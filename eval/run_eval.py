"""
检索质量评测脚本。

用途：给「检索」这一步单独打分。S2 的 chunk 策略对照实验靠它产出可比数字，
S5 会在此基础上扩展成完整评测体系（更多问题 + 生成侧指标）。

指标定义（都用「答案所在块能否被召回」来衡量，属于检索层指标）：
  Hit@k  : 前 k 个召回块里，是否存在包含任意期望关键词的块
  MRR    : 第一个命中块的排名倒数（第 1 位得 1.0，第 2 位得 0.5，未命中得 0）
  有效性问题：若全库没有任何块包含该题的关键词，说明题目或语料有问题，
              标记为 invalid 并从指标中剔除——避免拿一道「根本无答案」的题拉低分数

注意：关键词命中是 Hit Rate 的廉价实现（不需要人工标注「哪一块是标准答案」）。
它是代理指标，不是真值；结论只能横向比较（同一批题、同一套判定），不能当绝对分数。

用法：
  D:\\anaconda\\python.exe eval/run_eval.py --index data/vector_store/index.json --top-k 4
  D:\\anaconda\\python.exe eval/run_eval.py --questions eval/questions.json --detail
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# 保证能 import 到项目根目录下的 config / src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import VECTOR_STORE_DIR, RETRIEVAL_TOP_K  # noqa: E402
from src.embedding import get_embedding_model  # noqa: E402


def load_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        print(f"[ERROR] 索引文件不存在: {index_path}")
        print("        先执行: python main.py --recreate")
        sys.exit(1)
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[ERROR] 问题集不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def cosine_scores(index: Dict[str, Any], query_vec: np.ndarray) -> np.ndarray:
    """对所有块算与 query 的余弦相似度。等价于「暴力检索」，与线上 SimpleVectorStore 一致。"""
    doc_vecs = np.asarray(index["embeddings"], dtype=np.float32)
    # 预归一化能让余弦退化为点积；模型输出未归一化，这里显式归一化，避免除零
    doc_norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1e-12
    q_norm = np.linalg.norm(query_vec)
    q_norm = q_norm if q_norm > 0 else 1e-12
    return (doc_vecs / doc_norms) @ (query_vec / q_norm)


def corpus_keyword_coverage(index: Dict[str, Any], keywords: List[str]) -> int:
    """统计全库中有多少块包含该题的关键词，用于判断题目是否有效。"""
    count = 0
    for doc in index["documents"]:
        text = doc.get("page_content", "")
        if any(k in text for k in keywords):
            count += 1
    return count


def evaluate(
    index: Dict[str, Any],
    questions: List[Dict[str, Any]],
    embedding_model,
    top_k: int,
    detail: bool,
) -> Dict[str, Any]:
    n_chunks = len(index["documents"])
    dim = len(index["embeddings"][0]) if index["embeddings"] else 0

    print("=" * 78)
    print(f"语料块数: {n_chunks}    向量维度: {dim}    评测题目: {len(questions)}    Top-K: {top_k}")
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

        coverage = corpus_keyword_coverage(index, keywords) if keywords else 0
        query_vec = np.asarray(embedding_model.embed_query(question), dtype=np.float32)
        scores = cosine_scores(index, query_vec)

        # 取前 top_k（不设阈值截断，保证口径稳定：始终看前 k 名）
        order = np.argsort(scores)[::-1][:top_k]
        ranked = [(int(i), float(scores[i])) for i in order]

        first_hit_rank = None
        for rank, (idx, _score) in enumerate(ranked, start=1):
            text = index["documents"][idx].get("page_content", "")
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

        top1 = ranked[0] if ranked else (None, 0.0)
        top1_src = (
            Path(index["documents"][top1[0]].get("metadata", {}).get("source", "?")).name
            if top1[0] is not None else "-"
        )

        per_question.append({
            "id": qid,
            "question": question,
            "status": status,
            "first_hit_rank": first_hit_rank,
            "corpus_coverage": coverage,
            "top1_source": top1_src,
            "top1_score": round(top1[1], 4),
            "note": q.get("note", ""),
        })

        if detail:
            print(f"\n[{qid}] {status}  题目: {question}")
            print(f"      库中含关键词的块数: {coverage}   首个命中排名: {first_hit_rank}")
            for rank, (idx, score) in enumerate(ranked, start=1):
                meta = index["documents"][idx].get("metadata", {})
                src = Path(meta.get("source", "?")).name
                page = meta.get("page_label", meta.get("page", "?"))
                snippet = index["documents"][idx].get("page_content", "").replace("\n", " ")[:60]
                mark = "*" if (keywords and any(k in index["documents"][idx].get("page_content", "") for k in keywords)) else " "
                print(f"      {mark} #{rank} score={score:.4f}  {src} p{page}  {snippet}")

    elapsed = time.time() - t0
    summary = {
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
    parser.add_argument("--index", type=str, default=str(VECTOR_STORE_DIR / "index.json"),
                        help="索引文件路径")
    parser.add_argument("--questions", type=str, default=str(PROJECT_ROOT / "eval" / "questions.json"),
                        help="问题集 JSON 路径")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K, help="召回条数")
    parser.add_argument("--detail", action="store_true", help="逐题打印召回明细")
    parser.add_argument("--save", type=str, default="", help="把结果 JSON 存到指定路径")
    args = parser.parse_args()

    index = load_index(Path(args.index))
    questions = load_questions(Path(args.questions))
    embedding_model = get_embedding_model()

    result = evaluate(index, questions, embedding_model, args.top_k, args.detail)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
