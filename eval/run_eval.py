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

S4 改动：检索策略可切换，用于「混合检索 + 精排」的单变量对照实验：
  --retriever vector        纯向量（S3 基线，默认，与历史数字同口径）
  --retriever bm25          纯 BM25 稀疏检索
  --retriever hybrid        向量 + BM25，RRF 融合
  --rerank                  在所选策略之上加 Cross-Encoder 精排
四组分别对应实验 E7-1 ~ E7-4，每次都只改一个变量。

注意：关键词命中是 Hit Rate 的廉价实现（不需要人工标注「哪一块是标准答案」）。
它是代理指标，不是真值；结论只能横向比较（同一批题、同一套判定），不能当绝对分数。

S5 改动：支持分层题库（eval/questions_v2.json），按「检索难点」而不是按文档分层：
  single_fact  答案在单文档少量段落 —— 基本召回（Hit/MRR）
  distractor   术语跨文档出现但只有一个文档给答案 —— 来源命中率（关键词指标看不见）
  multi_hop    必须同时拿到两份文档 —— 来源覆盖率（required_sources）
  no_answer    语料没有该知识 —— 不计入检索指标，交给生成层测拒答
老题库 questions.json 没有 type 字段，按 single_fact 处理，保证历史数字可比。

用法：
  D:\\anaconda\\python.exe eval/run_eval.py --top-k 4
  D:\\anaconda\\python.exe eval/run_eval.py --backend faiss --faiss-index-type hnsw
  D:\\anaconda\\python.exe eval/run_eval.py --retriever hybrid --candidate-k 50
  D:\\anaconda\\python.exe eval/run_eval.py --retriever hybrid --rerank
  D:\\anaconda\\python.exe eval/run_eval.py --questions eval/questions_v2.json --retriever hybrid
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# 保证能 import 到项目根目录下的 config / src，以及同目录的 judge.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from config.config import (  # noqa: E402
    RERANK_ENABLED,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    VECTOR_BACKEND,
    VECTOR_STORE_DIR,
)
from src.embedding import get_embedding_model  # noqa: E402
from src.vector_store import load_vector_store  # noqa: E402


def build_retriever(store, mode: str, candidate_k: int, rrf_k: int, rerank: bool,
                    weights: str = ""):
    """
    组装检索器。四种策略只在「开了哪几路召回 + 要不要精排」上不同，
    其余（向量库、embedding 模型、候选数、评测题）全部固定 —— 单变量。

    实际组装交给 src.hybrid_retriever.create_hybrid_retriever，
    与 main.py 走同一份代码，避免评测口径与线上不一致。

    Args:
        weights: 形如 "1,2" -> vector:1.0, bm25:2.0；空串表示等权。
                 只影响多通道（hybrid）情形。

    Returns:
        (retriever, label)
    """
    from src.hybrid_retriever import create_hybrid_retriever

    channel_weights = None
    if weights:
        parts = [float(x) for x in weights.replace("，", ",").split(",") if x.strip()]
        # 通道顺序固定为 vector, bm25（与 HybridRetriever 内部 append 顺序一致）
        keys = [k for k, on in
                (("vector", mode in ("vector", "hybrid")),
                 ("bm25", mode in ("bm25", "hybrid"))) if on]
        if len(parts) != len(keys):
            raise ValueError(f"--rrf-weight 需要 {len(keys)} 个数（对应 {keys}），收到 {len(parts)}")
        channel_weights = dict(zip(keys, parts))

    retriever = create_hybrid_retriever(
        store, mode=mode, candidate_k=candidate_k, rrf_k=rrf_k, rerank=rerank,
        channel_weights=channel_weights,
    )
    label = mode + ("+rerank" if rerank else "")
    if channel_weights:
        label += "(" + ",".join(f"{k}={v:g}" for k, v in channel_weights.items()) + ")"
    return retriever, label


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
    retriever,
    label: str,
    judge=None,
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
        f"检索策略: {label}    评测题目: {len(questions)}    Top-K: {top_k}"
    )
    if judge is not None:
        print("命中判定: LLM-as-Judge（语义判定，带缓存）")
    print("=" * 78)

    per_question: List[Dict[str, Any]] = []
    hit_at_1 = hit_at_k = 0
    rr_sum = 0.0
    valid_count = 0
    distractor_count = source_hit_count = source_at1_count = 0
    hop_count = 0
    hop_cov_sum = 0.0
    hop_full = 0
    na_count = 0
    t0 = time.time()
    retrieve_sec = 0.0

    for q in questions:
        qid = q.get("id", "-")
        question = q["question"]
        keywords = q.get("expected_keywords", [])
        # v2 题库带 type；老题库没有，按 single_fact 处理，保证历史数据可比
        qtype = q.get("type", "single_fact")
        expected_source = q.get("expected_source")
        required_sources = q.get("required_sources") or []

        # 两套判定口径，二选一：
        #   keyword —— any(关键词 in 块文本)。便宜、确定，但系统性偏袒字面匹配（见 judge.py 说明）
        #   judge   —— LLM 语义判定。贵、慢，但不受用词影响
        # 多跳题与无答案题不走关键词判定：
        #   前者没有单一关键词（答案要靠两处拼），后者本来就没有答案。
        use_keyword = qtype in ("single_fact", "distractor")

        if judge is not None and use_keyword:
            coverage = -1   # judge 模式无法（也不该）对全库算覆盖率

            def is_hit(text, _q=question):
                return judge.is_relevant(_q, text)
        elif use_keyword:
            coverage = corpus_keyword_coverage(documents, keywords) if keywords else 0

            def is_hit(text, _kw=keywords):
                return bool(_kw) and any(k in text for k in _kw)
        else:
            coverage = -1
            is_hit = None

        # 走线上检索路径。return_debug 只做几个 dict 拼装，开销可忽略，
        # 但能让我们看清「这块是向量捞上来的还是 BM25 捞上来的」。
        t_ret = time.time()
        out = retriever.retrieve(
            question, embedding_model, top_k=top_k, return_debug=True
        )
        retrieve_sec += time.time() - t_ret

        ranked = [(doc, 0.0) for doc in out["documents"]]
        rows = out.get("debug", {}).get("rows", [])
        # 本次召回覆盖到哪些文档：多跳题与来源判定都基于它
        sources_in_topk = {Path(d.get("metadata", {}).get("source", "?")).name for d, _ in ranked}

        first_hit_rank = None
        if is_hit is not None:
            for rank, (doc, _score) in enumerate(ranked, start=1):
                if is_hit(doc.get("page_content", "")):
                    first_hit_rank = rank
                    break

        # 多跳：要求 top-k 同时覆盖所有 required_sources
        hop_coverage = None
        if qtype == "multi_hop" and required_sources:
            hop_coverage = len(set(required_sources) & sources_in_topk) / len(required_sources)

        source_hit = expected_source in sources_in_topk if expected_source else None
        top1_doc, _ = ranked[0] if ranked else (None, None)
        top1_src = (
            Path(top1_doc.get("metadata", {}).get("source", "?")).name
            if top1_doc is not None else "-"
        )
        source_at1 = (top1_src == expected_source) if expected_source else None

        if qtype == "no_answer":
            # 无答案题在检索层不计分：对「语料没有的东西」，捞到相似内容不算错
            status = "NO_ANSWER"
        elif qtype == "multi_hop":
            status = "HOP_ALL" if hop_coverage == 1.0 else f"HOP_{hop_coverage:.2f}"
        elif coverage == 0:
            status = "INVALID"      # 库里根本没有答案，题目无效
        elif first_hit_rank == 1:
            status = "HIT@1"
        elif first_hit_rank is not None:
            status = f"HIT@{first_hit_rank}"
        else:
            status = "MISS"

        if qtype == "no_answer":
            na_count += 1
        elif qtype == "multi_hop":
            hop_count += 1
            hop_cov_sum += hop_coverage or 0.0
            hop_full += 1 if hop_coverage == 1.0 else 0
        elif status != "INVALID":
            valid_count += 1
            hit_at_k += 1 if first_hit_rank else 0
            hit_at_1 += 1 if first_hit_rank == 1 else 0
            rr_sum += (1.0 / first_hit_rank) if first_hit_rank else 0.0
            if qtype == "distractor":
                distractor_count += 1
                source_hit_count += 1 if source_hit else 0
                source_at1_count += 1 if source_at1 else 0
        score_kind = out.get("debug", {}).get("score_kind", "score")
        top1_row = rows[0] if rows else {}
        top1_score = top1_row.get("rerank") if top1_row.get("rerank") is not None \
            else top1_row.get("fused", 0.0)

        per_question.append({
            "id": qid,
            "type": qtype,
            "question": question,
            "status": status,
            "first_hit_rank": first_hit_rank,
            "corpus_coverage": coverage,
            "sources_in_topk": sorted(sources_in_topk),
            "required_sources": required_sources,
            "source_hit": source_hit,
            "source_at1": source_at1,
            "hop_coverage": hop_coverage,
            "top1_source": top1_src,
            "top1_score": round(float(top1_score or 0.0), 4),
            "score_kind": score_kind,
            "note": q.get("note", ""),
        })

        if detail:
            print(f"\n[{qid}] {status}  ({qtype})  题目: {question}")
            extra = ""
            if qtype == "multi_hop":
                extra = f"  需要 {required_sources}  覆盖 {hop_coverage:.2f}"
            elif expected_source:
                extra = f"  期望来源 {expected_source}  来源命中={source_hit}"
            print(f"      库中含关键词的块数: {coverage}   首个命中排名: {first_hit_rank}{extra}")
            for rank, (doc, _) in enumerate(ranked, start=1):
                meta = doc.get("metadata", {})
                src = Path(meta.get("source", "?")).name
                page = meta.get("page_label", meta.get("page", "?"))
                snippet = doc.get("page_content", "").replace("\n", " ")[:60]
                if judge is not None and is_hit is not None:
                    hit, reason = judge.explain(question, doc.get("page_content", ""))
                    reason_txt = f"  ⟵判定: {reason}"
                elif is_hit is not None:
                    hit = is_hit(doc.get("page_content", ""))
                    reason_txt = ""
                else:
                    # 多跳/无答案题没有关键词判定，只标出来源
                    hit = src in (required_sources or [])
                    reason_txt = ""
                row = rows[rank - 1] if rank - 1 < len(rows) else {}
                # 每路名次：能看清「这块是向量捞的、BM25 捞的，还是被精排顶上来的」
                rk = row.get("ranks", {})
                rk_txt = " ".join(f"{ch}={rk[ch] if rk.get(ch) else '-'}" for ch in rk)
                if row.get("rerank") is not None:
                    score_txt = f"rerank={row['rerank']:.4f} rrf={row.get('fused', 0):.5f}"
                else:
                    score_txt = f"{score_kind}={row.get('fused', 0):.5f}"
                print(f"      {'*' if hit else ' '} #{rank} {score_txt} [{rk_txt}]  {src} p{page}  {snippet}{reason_txt}")

    elapsed = time.time() - t0
    type_counts = Counter(p["type"] for p in per_question)
    summary = {
        "backend": store.backend_name,
        "index_type": live_type,
        "retriever": label,
        "judge": "llm" if judge is not None else "keyword",
        "chunks": n_chunks,
        "dim": dim,
        "top_k": top_k,
        "questions_total": len(questions),
        "type_counts": dict(type_counts),
        "questions_valid": valid_count,
        "hit_at_1": hit_at_1,
        "hit_at_k": hit_at_k,
        "hit_rate_at_1": round(hit_at_1 / valid_count, 4) if valid_count else 0.0,
        "hit_rate_at_k": round(hit_at_k / valid_count, 4) if valid_count else 0.0,
        "mrr": round(rr_sum / valid_count, 4) if valid_count else 0.0,
        # 干扰题：答案所在文档是否出现在 top-k / top-1。关键词命中看不见「捞错书」
        "distractor_n": distractor_count,
        "source_hit_at_k": source_hit_count,
        "source_accuracy_at_k": round(source_hit_count / distractor_count, 4) if distractor_count else None,
        "source_accuracy_at_1": round(source_at1_count / distractor_count, 4) if distractor_count else None,
        # 多跳题：required_sources 的平均覆盖率与「全覆盖率」
        "multihop_n": hop_count,
        "multihop_coverage": round(hop_cov_sum / hop_count, 4) if hop_count else None,
        "multihop_full_rate": round(hop_full / hop_count, 4) if hop_count else None,
        "no_answer_n": na_count,
        # 平均单题检索延迟：混合检索与精排的代价就体现在这个数上
        "avg_retrieve_ms": round(retrieve_sec * 1000 / len(questions), 1) if questions else 0.0,
        "elapsed_sec": round(elapsed, 1),
    }

    print("\n" + "=" * 78)
    print(f"分层: {dict(type_counts)}   总耗时: {elapsed:.1f}s")
    print(f"可回答题: {valid_count}（Hit/MRR 只在这部分上算）  无答案题: {na_count}（不计入检索指标）")
    print(f"Hit@{1}  = {summary['hit_rate_at_1']:.4f}   ({hit_at_1}/{valid_count})")
    print(f"Hit@{top_k}  = {summary['hit_rate_at_k']:.4f}   ({hit_at_k}/{valid_count})")
    print(f"MRR    = {summary['mrr']:.4f}    平均检索延迟 = {summary['avg_retrieve_ms']:.1f} ms/题")
    if distractor_count:
        print(
            f"干扰题来源命中: @{top_k} = {summary['source_accuracy_at_k']:.4f} "
            f"({source_hit_count}/{distractor_count})   @1 = {summary['source_accuracy_at_1']:.4f} "
            f"({source_at1_count}/{distractor_count})"
        )
    if hop_count:
        print(
            f"多跳来源覆盖 = {summary['multihop_coverage']:.4f}  全覆盖率 = "
            f"{summary['multihop_full_rate']:.4f}  ({hop_full}/{hop_count})"
        )
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

    hop_partial = [p for p in per_question
                   if p["type"] == "multi_hop" and (p["hop_coverage"] or 0) < 1.0]
    if hop_partial:
        print("\n多跳未全覆盖（缺哪份文档）:")
        for p in hop_partial:
            missing = sorted(set(p.get("required_sources") or []) - set(p["sources_in_topk"]))
            print(f"  [{p['id']}] 覆盖={p['hop_coverage']:.2f} 缺={missing}  {p['question'][:40]}")

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
    parser.add_argument("--retriever", type=str, default="vector",
                        choices=["vector", "bm25", "hybrid"],
                        help="检索策略：vector（默认，S3 基线）| bm25 | hybrid（向量+BM25+RRF）")
    parser.add_argument("--candidate-k", type=int, default=RETRIEVAL_CANDIDATE_K,
                        help="每路召回的候选数（应远大于 --top-k）")
    parser.add_argument("--rrf-k", type=int, default=RRF_K, help="RRF 平滑常数")
    parser.add_argument("--rrf-weight", type=str, default="",
                        help="各通道 RRF 权重，如 \"1,2\"（vector:1, bm25:2）；默认等权")
    parser.add_argument("--rerank", action="store_true",
                        help="对融合候选做 Cross-Encoder 精排")
    parser.add_argument("--judge", action="store_true",
                        help="用 LLM 做语义判定替代关键词判定（第二套口径，需 API key）")
    parser.add_argument("--judge-cache", type=str,
                        default=str(PROJECT_ROOT / "results" / "judge_cache.json"),
                        help="LLM 判定缓存路径（必须保留，否则每次评测都重新调用且不可复现）")
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

    rerank = args.rerank or RERANK_ENABLED
    retriever, label = build_retriever(
        store, args.retriever, args.candidate_k, args.rrf_k, rerank, args.rrf_weight
    )

    # 预热：首次调用会触发 reranker 模型加载（数百 MB，数秒）与 BM25 建索引
    # （jieba 分词全语料，约 1~5 秒）。这些是**一次性启动成本**，
    # 不该混进「单题检索延迟」——否则延迟指标会被首题的开销严重污染。
    print("[INFO] 预热检索器（模型加载 / BM25 建索引，不计入延迟统计）")
    retriever.retrieve(questions[0]["question"], embedding_model, top_k=args.top_k)

    judge = None
    if args.judge:
        from judge import LLMJudge  # eval/ 目录本身在 sys.path 里

        judge = LLMJudge(Path(args.judge_cache))

    result = evaluate(
        store, questions, embedding_model, args.top_k, args.detail, retriever, label,
        judge=judge,
    )

    if judge is not None:
        judge.save()
        print(f"LLM 判定统计: {judge.stats()}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
