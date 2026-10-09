"""
检索质量评测脚本。

用途：给「检索」这一步单独打分。S2 的 chunk 策略对照实验靠它产出可比数字，
S5 会在此基础上扩展成完整评测体系（更多问题 + 生成侧指标）。

指标定义（都用「答案所在块能否被召回」来衡量，属于检索层指标）：
  Hit@k  : 前 k 个召回块里，是否存在包含任意期望关键词的块
  MRR    : 第一个命中块的排名倒数（第 1 位得 1.0，第 2 位得 0.5，未命中得 0）
  有效性问题：若全库没有任何块包含该题的关键词，说明题目或语料有问题，
              标记为 invalid 并从指标中剔除——避免拿一道「根本无答案」的题拉低分数

S3 改动：不再自己读 index.json 算余弦，而是**直接调用线上向量库**
（src.vector_store），这样评的就是真实检索路径。做 chunk 策略 A/B 时请固定
VECTOR_BACKEND=brute，保证与历史数字同口径（brute 是精确检索 = 真值）。

注意：关键词命中是 Hit Rate 的廉价实现（不需要人工标注「哪一块是标准答案」）。
它是代理指标，不是真值；结论只能横向比较（同一批题、同一套判定），不能当绝对分数。

用法：
  D:\\anaconda\\python.exe eval/run_eval.py --top-k 4
  D:\\anaconda\\python.exe eval/run_eval.py --backend faiss --faiss-index-type hnsw
  set VECTOR_BACKEND=brute && D:\\anaconda\\python.exe eval/run_eval.py --detail
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# 保证能 import 到项目根目录下的 config / src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import VECTOR_STORE_DIR, RETRIEVAL_TOP_K, VECTOR_BACKEND  # noqa: E402
from src.embedding import get_embedding_model  # noqa: E402
from src.vector_store import load_vector_store  # noqa: E402


def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[ERROR] 问题集不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def corpus_keyword_coverage(documents: List[Dict[str, Any]], keywords: List[str]) -> int:
    """统计全库中有多少块包含该题的关键词，用于判断题目是否有效。"""
    count = 0
    for doc in documents:
        text = doc.get("page_content", "")
        if any(k in text for k in keywords):
            count += 1
    return count


def evaluate(
    store,
    questions: List[Dict[str, Any]],
    embedding_model,
    top_k: int,
    detail: bool,
) -> Dict[str, Any]:
    documents = store.documents
    n_chunks = len(documents)
    dim = int(store.meta.get("dim") or 0)
    # 取实例上的实时类型：--faiss-index-type 会在内存里重建索引，
    # 此时磁盘 meta 记的类型（可能是上次落盘的）已经过时
    live_type = getattr(store, "index_type", None) or store.meta.get("index_type", "-")

    print("=" * 78)
    print(
        f"语料块数: {n_chunks}    向量维度: {dim}    "
        f"检索后端: {store.backend_name}/{live_type}    "
        f"评测题目: {len(questions)}    Top-K: {top_k}"
    )
    print("=" * 78)

    per_question: List[Dict[str, Any]] = []
    hit_at_1 = hit_at_k = 0
    rr_sum = 0.0
    valid_count = 0
    t0 = time.time()

    for q in questions:
        qid = q.get("id", "-")
        question = q["question"]
        keywords = q.get("expected_keywords", [])

        coverage = corpus_keyword_coverage(documents, keywords) if keywords else 0
        query_vec = embedding_model.embed_query(question)

        # 走线上检索路径；threshold 显式关掉，保证口径稳定：始终看前 k 名
        hits = store.similarity_search(
            query_vec, k=top_k, threshold=float("-inf"), with_scores=True
        )
        ranked = [(doc, float(score)) for doc, score in hits]

        first_hit_rank = None
        for rank, (doc, _score) in enumerate(ranked, start=1):
            text = doc.get("page_content", "")
            if keywords and any(k in text for k in keywords):
                first_hit_rank = rank
                break

        if coverage == 0:
            status = "INVALID"      # 库里根本没有答案，题目无效
        elif first_hit_rank == 1:
            status = "HIT@1"
        elif first_hit_rank is not None:
            status = f"HIT@{first_hit_rank}"
        else:
            status = "MISS"

        if status != "INVALID":
            valid_count += 1
            hit_at_k += 1 if first_hit_rank else 0
            hit_at_1 += 1 if first_hit_rank == 1 else 0
            rr_sum += (1.0 / first_hit_rank) if first_hit_rank else 0.0

        top1_doc, top1_score = ranked[0] if ranked else (None, 0.0)
        top1_src = (
            Path(top1_doc.get("metadata", {}).get("source", "?")).name
            if top1_doc is not None else "-"
        )

        per_question.append({
            "id": qid,
            "question": question,
            "status": status,
            "first_hit_rank": first_hit_rank,
            "corpus_coverage": coverage,
            "top1_source": top1_src,
            "top1_score": round(top1_score, 4),
            "note": q.get("note", ""),
        })

        if detail:
            print(f"\n[{qid}] {status}  题目: {question}")
            print(f"      库中含关键词的块数: {coverage}   首个命中排名: {first_hit_rank}")
            for rank, (doc, score) in enumerate(ranked, start=1):
                meta = doc.get("metadata", {})
                src = Path(meta.get("source", "?")).name
                page = meta.get("page_label", meta.get("page", "?"))
                snippet = doc.get("page_content", "").replace("\n", " ")[:60]
                hit = keywords and any(k in doc.get("page_content", "") for k in keywords)
                print(f"      {'*' if hit else ' '} #{rank} score={score:.4f}  {src} p{page}  {snippet}")

    elapsed = time.time() - t0
    summary = {
        "backend": store.backend_name,
        "index_type": live_type,
        "chunks": n_chunks,
        "dim": dim,
        "top_k": top_k,
        "questions_total": len(questions),
        "questions_valid": valid_count,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "hit_rate_at_1": round(hit_at_1 / valid_count, 4) if valid_count else 0.0,
        "hit_rate_at_k": round(hit_at_k / valid_count, 4) if valid_count else 0.0,
        "mrr": round(rr_sum / valid_count, 4) if valid_count else 0.0,
        "elapsed_sec": round(elapsed, 1),
    }

    print("\n" + "=" * 78)
    print(f"有效题目: {valid_count}/{len(questions)}   总耗时: {elapsed:.1f}s")
    print(f"Hit@{1}  = {summary['hit_rate_at_1']:.4f}   ({hit_at_1}/{valid_count})")
    print(f"Hit@{top_k}  = {summary['hit_rate_at_k']:.4f}   ({hit_at_k}/{valid_count})")
    print(f"MRR    = {summary['mrr']:.4f}")
    print("=" * 78)

    # 未命中题目单独列出，方便定位
    misses = [p for p in per_question if p["status"] == "MISS"]
    if misses:
        print("\n未命中题目:")
        for p in misses:
            print(f"  [{p['id']}] {p['question']}   (top1={p['top1_source']} score={p['top1_score']})")

    invalids = [p for p in per_question if p["status"] == "INVALID"]
    if invalids:
        print("\n无效题目（语料中找不到关键词，需检查题目或语料）:")
        for p in invalids:
            print(f"  [{p['id']}] {p['question']}")

    return {"summary": summary, "per_question": per_question}


def main():
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--store-dir", type=str, default=str(VECTOR_STORE_DIR),
                        help="向量库目录（含 documents.json / embeddings.npy）")
    parser.add_argument("--questions", type=str, default=str(PROJECT_ROOT / "eval" / "questions.json"),
                        help="问题集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K, help="召回条数")
    parser.add_argument("--backend", type=str, default=VECTOR_BACKEND,
                        help="检索后端：brute | faiss")
    parser.add_argument("--faiss-index-type", type=str, default=None,
                        help="FAISS 索引类型：flat | hnsw | ivf（仅 backend=faiss 生效）")
    parser.add_argument("--detail", action="store_true", help="逐题打印召回明细")
    parser.add_argument("--save", type=str, default="", help="把结果 JSON 存到指定路径")
    args = parser.parse_args()

    if args.faiss_index_type:
        print(f"[INFO] FAISS 索引类型覆盖为 {args.faiss_index_type}（内存中按需重建，不落盘）")

    questions = load_questions(Path(args.questions))
    embedding_model = get_embedding_model()

    store = load_vector_store(
        Path(args.store_dir), backend=args.backend, index_type=args.faiss_index_type
    )
    if store is None:
        print(f"[ERROR] 未找到向量库: {args.store_dir}")
        print("        先执行: python main.py --recreate")
        sys.exit(1)

    result = evaluate(store, questions, embedding_model, args.top_k, args.detail)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
