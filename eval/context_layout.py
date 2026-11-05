"""
上下文布局诊断：答案在送给模型的那段文本里，到底落在哪个位置。

为什么需要这个脚本
------------------
生成层指标（过度拒答 / 忠实度）只能告诉你「变差了几个题」，说不了为什么。
父块方案改变了上下文的**形状**：块变长、块变少、答案在块内的偏移后移。
只报指标不报形状，就会出现「调了参数、数字变了、不知道为什么」的情况 ——
这种结论下次换个语料就不成立。

这里量三个量，都是不需要调 LLM 的纯统计量：
  1. context 总字符数        —— 成本
  2. 关键词是否出现在 context —— 内容是否还在（召回有没有丢）
  3. 首次命中的**相对位置**   —— 模型要在多长的前缀之后才看到答案

第 3 项是关键。Transformer 对上下文的注意力不是均匀的，序列中部最容易被忽略
（"lost in the middle"）。相对位置从 0.13 涨到 0.27，含义是「答案被埋到更深处」，
这能解释「内容明明在、模型却说没找到」。

用法：
    python eval/context_layout.py --top-k 4
    python eval/context_layout.py --top-k 4 --parent-expand --parent-window 3
    python eval/context_layout.py --top-k 2 --parent-expand --parent-window 3
    # 一次跑多组做对照：
    python eval/context_layout.py --sweep
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding import get_embedding_model  # noqa: E402
from src.hybrid_retriever import create_hybrid_retriever  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import load_vector_store  # noqa: E402

from config.config import (  # noqa: E402
    PARENT_WINDOW,
    RETRIEVAL_CANDIDATE_K,
    RERANK_ENABLED,
    RETRIEVAL_TOP_K,
    VECTOR_STORE_DIR,
)

# _format_context 是实例方法，但它只用到 Path / 字符串拼接，不碰 self 的状态。
# 直接借过来用，而不是复制一份 —— 复制就会和线上拼接格式悄悄分叉，
# 测出来的位置和真实 prompt 里的位置对不上。
def _format(docs: List[Dict[str, Any]]) -> str:
    return RAGPipeline._format_context(None, docs)  # type: ignore[arg-type]


def layout_stats(retriever, questions: List[Dict[str, Any]], embedding_model) -> Dict[str, Any]:
    lengths: List[int] = []
    positions: List[float] = []
    hits = 0
    total = 0

    for q in questions:
        keywords = q.get("expected_keywords") or []
        # 无答案题本来就没有答案可定位，计入会把位置均值拉向随机
        if not keywords or q.get("type") == "no_answer":
            continue

        docs = retriever.retrieve(q["question"], embedding_model, top_k=retriever.top_k)
        context = _format(docs)
        lengths.append(len(context))
        total += 1

        # 取所有关键词里最靠前的那个位置：模型只要看到任一个可判定的锚点即可，
        # 取均值会被「同义但靠后」的词干扰。
        pos = None
        for kw in keywords:
            i = context.find(kw)
            if i >= 0:
                rel = i / len(context)
                pos = rel if pos is None else min(pos, rel)
        if pos is not None:
            hits += 1
            positions.append(pos)

    return {
        "n": total,
        "context_chars_avg": round(sum(lengths) / len(lengths)) if lengths else 0,
        "context_chars_max": max(lengths) if lengths else 0,
        "keyword_hit_rate": round(hits / total, 4) if total else 0.0,
        "first_hit_pos_avg": round(sum(positions) / len(positions), 4) if positions else None,
        "first_hit_pos_median": round(statistics.median(positions), 4) if positions else None,
        "first_hit_pos_max": round(max(positions), 4) if positions else None,
        # 落在后半段 = 模型要先读掉一半无关内容，是最容易漏答的那批题
        "second_half": sum(1 for p in positions if p > 0.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions_v2.json"))
    ap.add_argument("--store-dir", default=str(VECTOR_STORE_DIR))
    ap.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    ap.add_argument("--candidate-k", type=int, default=RETRIEVAL_CANDIDATE_K)
    ap.add_argument("--parent-expand", action="store_true")
    ap.add_argument("--parent-window", type=int, default=PARENT_WINDOW)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="跑内置的几组对照")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data

    embedding_model = get_embedding_model()
    store = load_vector_store(Path(args.store_dir))
    if store is None:
        print(f"[ERROR] 未找到向量库: {args.store_dir}")
        sys.exit(1)

    # (标签, top_k, 是否父块, 窗口)
    configs = [
        ("child k=4", 4, False, 3),
        ("parent w=3 k=4", 4, True, 3),
        ("parent w=3 k=2", 2, True, 3),
    ] if args.sweep else [
        (
            f"{'parent' if args.parent_expand else 'child'} "
            f"{'w=%d ' % args.parent_window if args.parent_expand else ''}k={args.top_k}",
            args.top_k,
            args.parent_expand,
            args.parent_window,
        )
    ]

    print("=" * 78)
    print(f"{'配置':<18}{'n':>4}{'上下文均值':>10}{'最大':>7}{'关键词命中':>10}{'首命中位置':>10}{'后半段':>8}")
    print("-" * 78)
    results = {}
    for label, k, expand, window in configs:
        r = create_hybrid_retriever(
            store,
            mode="hybrid",
            top_k=k,
            candidate_k=args.candidate_k,
            rerank=(not args.no_rerank) and RERANK_ENABLED,
            parent_expand=expand,
            parent_window=window,
        )
        s = layout_stats(r, questions, embedding_model)
        results[label] = s
        print(
            f"{label:<18}{s['n']:>4}{s['context_chars_avg']:>10}{s['context_chars_max']:>7}"
            f"{s['keyword_hit_rate']:>10.3f}{(s['first_hit_pos_avg'] or 0):>10.3f}"
            f"{s['second_half']:>8}"
        )
    print("=" * 78)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
