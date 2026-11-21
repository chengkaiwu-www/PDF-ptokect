"""
逐块查看检索结果与判定（Precision@k 的取证工具）。

为什么需要这个脚本
------------------
`Precision@k` 告诉你「k 块里有几块相关」，但**不告诉你被扣掉的是哪种块**。
一个数字可以有很多种解释：
  · 那几块是彻底的无关内容（噪声）
  · 那几块是同一本书的相邻段落（冗余但不有害）
  · 那几块其实能答，是判定器判错了（假阴性）
只有把块打印出来看，才能区分。**汇总指标永远需要抽查作为证据。**

用法：
    D:\\anaconda\\python.exe scripts/inspect_blocks.py --qid s04 --retriever vector
    D:\\anaconda\\python.exe scripts/inspect_blocks.py --qid s10 d11 --retriever hybrid --rerank
    D:\\anaconda\\python.exe scripts/inspect_blocks.py --qid s16 --no-judge
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from config.config import (  # noqa: E402
    RETRIEVAL_CANDIDATE_K,
    RRF_K,
    VECTOR_BACKEND,
    VECTOR_STORE_DIR,
)
from src.embedding import get_embedding_model  # noqa: E402
from src.vector_store import load_vector_store  # noqa: E402
from eval.run_eval import build_retriever  # noqa: E402


def snippet(text: str, n: int = 90) -> str:
    flat = " ".join((text or "").split())
    return flat[:n] + ("…" if len(flat) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser(description="逐块查看检索结果与相关性判定")
    ap.add_argument("--qid", nargs="+", required=True, help="题目 id，可多个")
    ap.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions_v2.json"))
    ap.add_argument("--retriever", default="hybrid", choices=["vector", "bm25", "hybrid"])
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--store-dir", default=str(VECTOR_STORE_DIR))
    ap.add_argument("--backend", default=VECTOR_BACKEND)
    ap.add_argument("--no-judge", action="store_true", help="不调用 LLM，只打印块")
    ap.add_argument("--judge-cache", default=str(PROJECT_ROOT / "results" / "judge_cache.json"))
    args = ap.parse_args()

    raw = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in (raw if isinstance(raw, list) else raw.get("questions", []))}

    embedding_model = get_embedding_model()
    store = load_vector_store(Path(args.store_dir), backend=args.backend)
    retriever, label = build_retriever(
        store, args.retriever, RETRIEVAL_CANDIDATE_K, RRF_K, args.rerank
    )

    judge = None
    if not args.no_judge:
        from judge import LLMJudge

        judge = LLMJudge(Path(args.judge_cache))

    print(f"检索策略: {label}    top_k={args.top_k}\n")
    for qid in args.qid:
        q = qs.get(qid)
        if not q:
            print(f"[{qid}] 题库里没有这道题\n")
            continue
        print("=" * 78)
        print(f"[{qid}] ({q.get('type')})  {q['question']}")
        if q.get("expected_source"):
            print(f"  期望来源: {q['expected_source']}")
        print("=" * 78)
        out = retriever.retrieve(
            q["question"], embedding_model, top_k=args.top_k, return_debug=True
        )
        for rank, doc in enumerate(out["documents"], start=1):
            meta = doc.get("metadata", {}) or {}
            text = doc.get("page_content", "")
            src = Path(meta.get("source", "?")).name
            page = meta.get("page_label") or meta.get("page", "?")
            flag = ""
            if judge is not None:
                flag = "相关  " if judge.is_relevant(q["question"], text) else "不相关"
            print(f"  #{rank} [{flag}] {src} p{page}")
            print(f"      {snippet(text)}")
        print()

    if judge is not None:
        judge.save()
        print(f"判定统计: {judge.stats()}")


if __name__ == "__main__":
    main()
