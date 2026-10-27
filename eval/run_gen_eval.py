"""
生成层评测：答案可不可信、该不该拒答。

## 为什么检索指标不够

S4 结束时语义口径下四策略 Hit@4 全是 1.000 —— 检索层已经饱和。
但「块捞对了」不等于「答案可信」，这中间还隔着三件事：

  1. **模型会不会编**（faithfulness）：块里有 A，模型答出 A+B，B 是它自己补的；
  2. **引用是不是真的**（citation validity）：答案标了页码，但那一页根本不在召回里；
  3. **该拒答时会不会硬答**（refusal）：语料里没有这个知识，模型凭参数记忆答了出来。

这三件事都是**确定性可测**的（引用校验甚至不需要模型），比再调检索参数有用得多。

## 指标定义

| 指标 | 计算方式 | 看什么 |
|---|---|---|
| 引用覆盖率 | 有 ≥1 个合法引用的答案 / 非拒答答案 | 愿不愿意标来源 |
| 引用合法率 | 合法引用数 / 引用总数 | 有没有编造页码 |
| 编造答案数 | 含 ≥1 个非法引用的答案数 | 绝对量，比比率更直观 |
| 忠实度 | 被资料支持的断言数 / 断言总数（LLM 判定） | 有没有超出资料 |
| 拒答率（无答案题） | 正确拒答数 / 无答案题数 | 该拒时是否拒 |
| 过度拒答率（有答案题） | 错误拒答数 / 有答案题数 | 不该拒时是否拒 |
| 多来源引用率（多跳题） | 引用了 ≥2 份文档的占比 | 多跳是否真的用了两份资料 |

**注意「忠实度」和「正确性」不是一回事**：忠实度只问「答案是否被资料支撑」。
资料本身错的、或检索给错块，答案可以「忠实但错误」。本项目不测正确性——
那需要人工标注参考答案，20~50 题的规模下标注质量不可控，测了也不可信。

用法：
    set HF_HUB_OFFLINE=1
    D:\\anaconda\\python.exe eval/run_gen_eval.py --questions eval/questions_v2.json --judge
    D:\\anaconda\\python.exe eval/run_gen_eval.py --retriever hybrid --rerank --detail
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from config.config import (  # noqa: E402
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    RERANK_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_TOP_K,
    RRF_K,
    VECTOR_STORE_DIR,
)
from src.embedding import get_embedding_model  # noqa: E402
from src.hybrid_retriever import create_hybrid_retriever  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import load_vector_store  # noqa: E402


def build_llm():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，生成层评测无法进行")
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        temperature=0,   # 评测要可复现；创意性不在这个项目的目标里
    )


def pct(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


# ---------------------------------------------------------------------------
# 独立的拒答判定
#
# 为什么不直接用 RAGPipeline 返回的 refused：
#   那是**哨兵匹配**的结果，而哨兵是 prompt 条款的产物。做 prompt 消融时
#   （去掉拒答条款）哨兵根本不会出现，于是「拒答率 0%」既可能是模型真的没拒，
#   也可能是它用自然语言拒了但不再输出哨兵 —— 两种情况混在一起，实验就废了。
#
# 所以评测侧必须用一套**与 prompt 无关**的判定：自然语言拒答模式 + 哨兵。
# 这是评测的一条通用原则：**度量工具不能依赖被测对象的输出格式**。
# ---------------------------------------------------------------------------
REFUSAL_PATTERNS = (
    "insufficient_context",
    "没有找到足够",
    "资料中没有",
    "资料未提及",
    "资料未说明",
    "未说明",
    "未给出",
    "没有提及",
    "未提及",
    "未提到",
    "没有提到",
    "无法回答",
    "无法从提供",
    "不足以回答",
    "没有相关信息",
    "不涉及",
)


def refusal_state(text: str) -> tuple:
    """
    判定 (是否拒答, 是否含"局部缺资料"表述)。

    第一版判定器有个真实 bug：只要出现"无法确定""未提及"就算拒答，
    结果把一条 169 字的正常答案判成拒答（正文里恰好写了"无法确定"）。
    改成**看位置和长度**：

      · 开头就是拒答话术        -> 拒答（这是模型第一反应）
      · 很短（<100 字）且提到缺资料 -> 拒答（短答复 + 缺资料 ≈ 整句就是拒答）
      · 长答案中间夹一句"资料未说明X，但Y…" -> 算作答，另记 partial_gap
        （它是**部分作答**，不是拒答；混为一谈会把"答案不完整"错记成"不敢答"）
    """
    t = (text or "").strip()
    low = t.lower()
    has_gap = any(p in low for p in REFUSAL_PATTERNS)

    if "insufficient_context" in t[:40].lower():
        return True, False
    if has_gap and len(t) < 100:
        return True, False
    if any(p in t[:30].lower() for p in REFUSAL_PATTERNS):
        return True, has_gap
    return False, has_gap


def evaluate(
    pipeline: RAGPipeline,
    questions: List[Dict[str, Any]],
    judge,
    detail: bool = False,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    gen_sec = 0.0

    for q in questions:
        qid = q["id"]
        qtype = q.get("type", "single_fact")
        question = q["question"]
        expect_refusal = (qtype == "no_answer")

        t0 = time.time()
        out = pipeline.invoke({"question": question})
        gen_sec += time.time() - t0

        sentinel = bool(out["refused"])
        # 评测侧独立判定：哨兵 + 自然语言模式，避免 prompt 变体之间口径漂移
        refused_eval, partial_gap = refusal_state(out.get("raw_answer", ""))
        refused_eval = refused_eval or sentinel
        citations = out["citations"]
        raw_answer = out.get("raw_answer", "")

        # 拒答的答案没有断言可判，忠实度不适用（返回 None 而不是 1.0）
        faith = None
        if not refused_eval and judge is not None:
            faith = judge.faithfulness(raw_answer, out["context"])

        row = {
            "id": qid,
            "type": qtype,
            "question": question,
            "refused": refused_eval,
            "sentinel_refused": sentinel,
            "partial_gap": partial_gap,
            "expect_refusal": expect_refusal,
            "refusal_correct": refused_eval == expect_refusal,
            "citation_count": citations["count"],
            "citation_valid": citations["valid"],
            "citation_invalid": len(citations["invalid"]),
            "cited_sources": citations["cited_sources"],
            "faithfulness": faith["score"] if faith else None,
            "faith_claims": faith["total"] if faith else 0,
            "faith_parse_failed": bool(faith and faith["parse_failed"]),
            "answer_chars": len(raw_answer),
            "answer": out["answer"],
        }
        rows.append(row)

        if detail:
            mark = "OK " if row["refusal_correct"] else "!! "
            cite_txt = (
                f"引用 {citations['valid']}/{citations['count']}"
                if citations["count"] else "无引用"
            )
            faith_txt = (
                f" 忠实度={faith['score'] if faith['score'] is not None else 'N/A'}"
                f"({faith['supported']}/{faith['total']})" if faith else ""
            )
            print(f"\n[{qid}] {mark}{qtype}  {'拒答' if refused else '作答'}  {cite_txt}{faith_txt}")
            print(f"      Q: {question}")
            print(f"      A: {out['answer'].replace(chr(10), ' ')[:160]}")
            if citations["invalid"]:
                print(f"      ⚠ 非法引用: {citations['invalid'][:3]}")

    # ---------------- 聚合 ----------------
    answerable = [r for r in rows if not r["expect_refusal"]]
    negatives = [r for r in rows if r["expect_refusal"]]
    answered = [r for r in answerable if not r["refused"]]

    cite_total = sum(r["citation_count"] for r in answered)
    cite_valid = sum(r["citation_valid"] for r in answered)
    with_cite = sum(1 for r in answered if r["citation_count"] > 0)
    fabricated = sum(1 for r in answered if r["citation_invalid"] > 0)

    faith_rows = [r for r in answered if r["faithfulness"] is not None]
    multi = [r for r in answerable if r["type"] == "multi_hop" and not r["refused"]]
    multi_ok = [r for r in multi if len(r["cited_sources"]) >= 2]

    summary = {
        "prompt_version": pipeline.prompt_version,
        "prompt_variant": pipeline.prompt_variant,
        "retriever": getattr(pipeline.retriever, "label", "-"),
        "judge": judge is not None,
        "questions_total": len(rows),
        "answerable": len(answerable),
        "answered": len(answered),
        "no_answer": len(negatives),
        # 引用层
        "answers_with_citation": with_cite,
        "citation_coverage": pct(with_cite, len(answered)),
        "citations_total": cite_total,
        "citations_valid": cite_valid,
        "citation_validity": pct(cite_valid, cite_total),
        "answers_with_fake_citation": fabricated,
        # 忠实度
        "faithfulness": round(
            sum(r["faithfulness"] for r in faith_rows) / len(faith_rows), 4
        ) if faith_rows else None,
        "faithfulness_answers": len(faith_rows),
        "faith_parse_failed": sum(1 for r in rows if r["faith_parse_failed"]),
        # 哨兵判定与评测侧判定的分歧数：不为 0 说明「拒答口径」本身需要讨论
        "sentinel_disagreements": sum(
            1 for r in rows if r["refused"] != r["sentinel_refused"]
        ),
        # 拒答层
        "refusal_rate_on_no_answer": pct(sum(1 for r in negatives if r["refused"]), len(negatives)),
        "over_refusal_rate": pct(sum(1 for r in answerable if r["refused"]), len(answerable)),
        # 多跳
        "multi_hop_questions": len(multi),
        "multi_hop_multi_source": len(multi_ok),
        "multi_hop_multi_source_rate": pct(len(multi_ok), len(multi)),
        "avg_gen_ms": round(gen_sec * 1000 / len(rows), 1) if rows else 0.0,
    }

    # 按类型分组的拒答/引用表现，避免整体平均掩盖某一层的问题
    by_type: Dict[str, Dict[str, Any]] = {}
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        by_type[t] = {
            "n": len(group),
            "refusal_correct": sum(1 for r in group if r["refusal_correct"]),
            "avg_citation_validity": round(
                sum(r["citation_valid"] for r in group) / max(1, sum(r["citation_count"] for r in group)), 4
            ),
            "avg_faithfulness": round(
                sum(r["faithfulness"] for r in group if r["faithfulness"] is not None)
                / max(1, sum(1 for r in group if r["faithfulness"] is not None)), 4
            ) if any(r["faithfulness"] is not None for r in group) else None,
        }

    return {"summary": summary, "by_type": by_type, "per_question": rows}


def main():
    ap = argparse.ArgumentParser(description="生成层评测（引用 / 忠实度 / 拒答）")
    ap.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions_v2.json"))
    ap.add_argument("--store-dir", default=str(VECTOR_STORE_DIR))
    ap.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    ap.add_argument("--retriever", default="hybrid", choices=["vector", "bm25", "hybrid"])
    ap.add_argument("--candidate-k", type=int, default=RETRIEVAL_CANDIDATE_K)
    ap.add_argument("--rrf-k", type=int, default=RRF_K)
    ap.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=RERANK_ENABLED)
    ap.add_argument("--judge", action="store_true", help="启用 LLM 忠实度判定（需 API key）")
    ap.add_argument("--judge-cache", default=str(PROJECT_ROOT / "results" / "judge_cache.json"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--only-type", default="", help="只跑某一层：single_fact/distractor/multi_hop/no_answer")
    ap.add_argument("--prompt-variant", default="full", choices=["full", "no_refusal", "no_cite"],
                    help="prompt 消融变体（full 为正式配置）")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--save", default="", help="结果 JSON 路径（默认 results/gen_eval_<variant>.json）")
    args = ap.parse_args()
    if not args.save:
        args.save = str(PROJECT_ROOT / "results" / f"gen_eval_{args.prompt_variant}.json")

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    if args.only_type:
        questions = [q for q in questions if q.get("type") == args.only_type]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("[ERROR] 筛选后没有题目")
        sys.exit(1)

    embedding_model = get_embedding_model()
    store = load_vector_store(Path(args.store_dir))
    if store is None:
        print(f"[ERROR] 未找到向量库: {args.store_dir}（先执行 main.py --reindex）")
        sys.exit(1)

    retriever = create_hybrid_retriever(
        store,
        mode=args.retriever,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        rerank=args.rerank,
        reranker_model=RERANKER_MODEL,
        top_k=args.top_k,
    )
    retriever.label = args.retriever + ("+rerank" if args.rerank else "")

    pipeline = RAGPipeline(
        llm=build_llm(), retriever=retriever, embedding_model=embedding_model,
        prompt_variant=args.prompt_variant,
    )

    judge = None
    if args.judge:
        from judge import LLMJudge

        judge = LLMJudge(Path(args.judge_cache))

    print("=" * 78)
    print(
        f"生成层评测    检索策略={retriever.label}  top_k={args.top_k}  "
        f"prompt={pipeline.prompt_version}  忠实度判定={'开' if judge else '关'}  题目={len(questions)}"
    )
    print("=" * 78)

    result = evaluate(pipeline, questions, judge, detail=args.detail)

    if judge is not None:
        judge.save()
        print(f"\n判定统计: {judge.stats()}")

    s = result["summary"]
    print("\n" + "=" * 78)
    print(f"有答案题 {s['answerable']}    无答案题 {s['no_answer']}")
    print(f"引用覆盖率   = {s['citation_coverage']:.4f}   ({s['answers_with_citation']}/{s['answered']} 个作答)")
    print(f"引用合法率   = {s['citation_validity']:.4f}   ({s['citations_valid']}/{s['citations_total']})")
    print(f"编造引用的答案 = {s['answers_with_fake_citation']} 个")
    if s["faithfulness"] is not None:
        print(f"忠实度       = {s['faithfulness']:.4f}   （{s['faithfulness_answers']} 个答案参与，解析失败 {s['faith_parse_failed']}）")
    print(f"无答案题拒答率 = {s['refusal_rate_on_no_answer']:.4f}   过度拒答率 = {s['over_refusal_rate']:.4f}")
    if s["multi_hop_questions"]:
        print(f"多跳多来源引用 = {s['multi_hop_multi_source_rate']:.4f}   ({s['multi_hop_multi_source']}/{s['multi_hop_questions']})")
    print(f"平均生成延迟   = {s['avg_gen_ms']:.0f} ms/题")
    if s["sentinel_disagreements"]:
        print(f"⚠ 哨兵与评测侧拒答判定分歧: {s['sentinel_disagreements']} 题（口径需人工确认）")
    print("=" * 78)

    print("\n按类型:")
    for t, v in result["by_type"].items():
        faith = f"{v['avg_faithfulness']:.3f}" if v["avg_faithfulness"] is not None else "N/A"
        print(f"  {t:12s} n={v['n']:2d}  拒答判定正确={v['refusal_correct']:2d}  "
              f"引用合法率={v['avg_citation_validity']:.3f}  忠实度={faith}")

    bad = [r for r in result["per_question"] if not r["refusal_correct"]]
    if bad:
        print("\n拒答判定错误:")
        for r in bad:
            kind = "该拒未拒" if r["expect_refusal"] else "不该拒却拒"
            print(f"  [{r['id']}] {kind}: {r['question'][:40]}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
