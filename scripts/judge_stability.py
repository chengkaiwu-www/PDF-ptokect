"""
判定器自身的信度：同一个 (问题, 块) 判两次，结论一样吗？

为什么必须测
------------
前面所有语义口径的数字（Hit@1 0.9143、Precision@4 0.8786、融合 +2 题……）
都建立在一个从未验证的前提上：**同一个块，判定器今天和明天给出同一个答案**。

`temperature=0` 只是「尽量确定」，不是「保证确定」：
  · 浮点累加的批次不同 → logits 末位不同 → 在概率接近的候选上可能翻转；
  · MoE / 服务端批处理 / 模型版本滚动更新，都不是我们能控制的；
  · 二分类边界上的样本（"只沾一点边"）本来就容易抖。
一旦抖动存在，「+2 题」这种量级的结论就可能是抖出来的。

三个指标，缺一不可
------------------
1. `一致率`：两次判定相同的比例。**单独看会骗人** —— 若 86% 的样本都是「相关」，
   一个永远回答「相关」的常量判定器也能拿到 0.86。
2. `Cohen's κ`：扣掉「偶然一致」后的真实一致性。
   κ = (p_o − p_e) / (1 − p_e)，p_e 由两次判定的边际分布算出。
   经验刻度：<0.4 差、0.4~0.6 中、0.6~0.8 好、>0.8 很好。
3. `Hit@1 抖动`：**真正关心的量**。一致率 0.95 若全落在不影响首命中的位置，
   对结论毫无影响；相反，20 个翻转里只要 3 个动了第 1 位，Hit@1 就会变。
   所以必须把「判定抖动」换算成「指标抖动」。

为什么还要分正负样本各自的一致率
--------------------------------
正类（相关）和负类（不相关）的判定难度不对称 —— 负类是一句「讲的完全是另一件事」，
正类是「沾一点边也算」。混在一起算总一致率，会被占比大的那一类主导。

用法
----
    $env:HF_HUB_OFFLINE="1"
    D:\\anaconda\\python.exe -u scripts\\judge_stability.py --n 20 --repeats 2 \\
        --configs hybrid vector --out results\\_stability.txt

成本：唯一 (问题, 块) 对数 × repeats 次调用，约 0.1~0.2 元。
禁止并发：限流失败会被降级成「不相关」，那会伪装成「判定不一致」。
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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


# --------------------------------------------------------------------- 统计

def cohens_kappa(pairs: List[tuple]) -> float:
    """
    两次判定的 Cohen's κ。

    p_e 的含义：如果两次判定**各自独立地**按自己的边际分布随机作答，
    碰巧一致的概率。κ 把这一部分从 p_o 里扣掉。
    """
    n = len(pairs)
    if not n:
        return 0.0
    p_o = sum(1 for a, b in pairs if a == b) / n
    p_a = sum(1 for a, _ in pairs if a) / n
    p_b = sum(1 for _, b in pairs if b) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_e >= 1.0:
        # 两次判定全是同一个值 → 无法区分「真一致」和「常量判定器」
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def rate(k: int, n: int) -> float:
    return round(k / n, 4) if n else 0.0


def majority(vs: List[bool]) -> bool:
    """
    严格过半投票。偶数次平局时判 False（保守）—— 与「判定失败降级为不相关」同一取向：
    宁可少算命中，也不要把「不确定」算成「相关」而虚高指标。
    """
    return sum(1 for v in vs if v) * 2 > len(vs)


def first_hit(blocks: List[Dict[str, Any]], idx: int) -> Optional[int]:
    """第 idx 次判定下，第一个被判相关的块的名次；全不相关则 None。"""
    for b in sorted(blocks, key=lambda x: x["rank"]):
        if b["verdicts"][idx]:
            return b["rank"]
    return None


def first_hit_vote(blocks: List[Dict[str, Any]]) -> Optional[int]:
    """多数票下的首命中名次。"""
    for b in sorted(blocks, key=lambda x: x["rank"]):
        if majority(b["verdicts"]):
            return b["rank"]
    return None


# --------------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="判定器稳定性：同输入重复判定，量一致性")
    ap.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions_v2.json"))
    ap.add_argument("--n", type=int, default=20, help="抽样题数（按 type 分层）")
    ap.add_argument("--repeats", type=int, default=2, help="每个 (问题,块) 判几次")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--configs", nargs="+", default=["hybrid", "vector"],
                    help="用哪几个检索配置的召回块（块池越杂，负样本越多）")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--store-dir", default=str(VECTOR_STORE_DIR))
    ap.add_argument("--backend", default=VECTOR_BACKEND)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--max-pairs", type=int, default=160, help="唯一对上限，防跑飞")
    ap.add_argument("--save", default=str(PROJECT_ROOT / "results" / "judge_stability.json"))
    ap.add_argument("--out", default=None, help="把报告同时写成 UTF-8 文本")
    args = ap.parse_args()

    lines: List[str] = []

    def emit(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    # ---------- 一、抽样（按题型分层，保证各类型都有代表）----------
    raw = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qs = raw if isinstance(raw, list) else raw.get("questions", [])
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for q in qs:
        by_type[str(q.get("type", "?"))].append(q)

    rng = random.Random(args.seed)
    per = max(1, args.n // max(len(by_type), 1))
    picked: List[dict] = []
    for t in sorted(by_type):
        pool = by_type[t]
        picked.extend(rng.sample(pool, min(per, len(pool))))
    picked = picked[: args.n]
    picked.sort(key=lambda q: q["id"])

    emit("=" * 78)
    emit(f"判定器稳定性  |  题 {len(picked)} 道 × 配置 {args.configs} × top_k={args.top_k}")
    emit(f"抽样（seed={args.seed}）：{', '.join(q['id'] for q in picked)}")
    emit(f"题型分布：{dict(Counter(q.get('type', '?') for q in picked))}")
    emit("=" * 78)

    # ---------- 二、检索：块池要杂，负样本才够 ----------
    emb = get_embedding_model()
    store = load_vector_store(Path(args.store_dir), backend=args.backend)

    blocks: List[Dict[str, Any]] = []   # 保留「哪个配置的第几名」，不做跨配置去重
    for cfg in args.configs:
        retriever, label = build_retriever(
            store, cfg, RETRIEVAL_CANDIDATE_K, RRF_K, args.rerank
        )
        emit(f"\n检索：{cfg} -> {label}")
        for q in picked:
            out = retriever.retrieve(
                q["question"], emb, top_k=args.top_k, return_debug=True
            )
            docs = out["documents"] if isinstance(out, dict) else out
            for rank, doc in enumerate(docs, start=1):
                text = doc.get("page_content", "") or ""
                meta = doc.get("metadata", {}) or {}
                blocks.append({
                    "qid": q["id"],
                    "qtype": q.get("type", "?"),
                    "cfg": cfg,
                    "rank": rank,
                    "question": q["question"],
                    "text": text,
                    "source": Path(meta.get("source", "?")).name,
                    "page": meta.get("page_label") or meta.get("page"),
                })

    # 去重只减少调用，不减少结构：同一个 (问题, 块) 只判一次，结果被多处引用。
    uniq: Dict[tuple, Dict[str, Any]] = {}
    for b in blocks:
        key = (b["question"], b["text"])
        uniq.setdefault(key, b)

    # 超过预算时**按题截断**，不是按对截断 —— 把一道题的 4 块砍掉 2 块，
    # 首命中位置就没法算了（Hit@1 需要完整的 top_k 序列）。
    if len(uniq) > args.max_pairs:
        keys_per_q: Dict[str, set] = defaultdict(set)
        for b in blocks:
            keys_per_q[b["qid"]].add((b["question"], b["text"]))
        kept_q: List[str] = []
        total = 0
        for q in picked:
            k = len(keys_per_q[q["id"]])
            if total + k > args.max_pairs:
                break
            total += k
            kept_q.append(q["id"])
        picked = [q for q in picked if q["id"] in kept_q]
        blocks = [b for b in blocks if b["qid"] in kept_q]
        uniq = {}
        for b in blocks:
            uniq.setdefault((b["question"], b["text"]), b)
        emit(f"\n预算 {args.max_pairs} 对 → 按题截到 {len(kept_q)} 道题（块保持完整）")

    uniq_list = list(uniq.values())

    emit(f"\n召回位置 {len(blocks)} 个 → 唯一 (问题,块) 对 {len(uniq_list)} 个"
         f"（去重省下 {len(blocks) - len(uniq_list)} 次重复判定）")
    emit(f"预计调用 {len(uniq_list) * args.repeats} 次（串行，避免限流伪装成不一致）")

    # ---------- 三、重复判定（绕过缓存）----------
    from judge import LLMJudge

    judge = LLMJudge(
        cache_path=None,          # 不落盘，别把随机样本写进正式缓存
        provider=args.judge_provider,
        model=args.judge_model,
    )
    emit(f"判定器：{judge.model_id}")

    t0 = time.time()
    verdict_of: Dict[tuple, List[bool]] = {}
    reason_of: Dict[tuple, List[str]] = {}
    for i, b in enumerate(uniq_list, start=1):
        key = (b["question"], b["text"])
        vs, rs = [], []
        for _ in range(args.repeats):
            v, r = judge.explain(b["question"], b["text"], bypass_cache=True)
            vs.append(v)
            rs.append(r)
        verdict_of[key] = vs
        reason_of[key] = rs
        tag = "".join("T" if v else "F" for v in vs)
        emit(f"  [{i:>3}/{len(uniq_list)}] {b['qid']:<3} {b['cfg']:<6} #{b['rank']} {tag}  {b['source']}")

    elapsed = time.time() - t0

    for b in blocks:
        key = (b["question"], b["text"])
        b["verdicts"] = verdict_of[key]
        b["reasons"] = reason_of[key]

    # ---------- 四、一致率 / κ ----------
    pairs = [(v[0], v[1]) for v in verdict_of.values()]   # 只看前两次，与 repeats=2 可比
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    f2t = sum(1 for a, b in pairs if not a and b)
    t2f = sum(1 for a, b in pairs if a and not b)
    pos = [(a, b) for a, b in pairs if a or b]      # 至少一次判相关
    neg = [(a, b) for a, b in pairs if not (a or b)]  # 两次都不相关

    emit("\n" + "=" * 78)
    emit(f"一、一致性（取前 2 次判定配对，与 repeats=2 口径可比）")
    emit("=" * 78)
    emit(f"  判两次一致     : {agree}/{n} = {rate(agree, n):.4f}")
    emit(f"  Cohen's κ      : {cohens_kappa(pairs):.4f}   "
         f"(>0.8 很好 / 0.6~0.8 好 / 0.4~0.6 中 / <0.4 差)")
    emit(f"  翻转           : 不相关→相关 {f2t}  |  相关→不相关 {t2f}")
    emit(f"  正类样本（至少一次相关）: {len(pos)} 个，一致 {sum(1 for a,b in pos if a==b)} "
         f"= {rate(sum(1 for a,b in pos if a==b), len(pos)):.4f}")
    emit(f"  负类样本（两次都不相关）: {len(neg)} 个，一致 {sum(1 for a,b in neg if a==b)} "
         f"= {rate(sum(1 for a,b in neg if a==b), len(neg)):.4f}")
    emit(f"  正类占比       : {rate(len(pos), n):.4f}"
         f"   ← 占比高时「一致率」会被常量判定器白送")

    # ---------- 四之二、多次投票：边界带到底有多宽 ----------
    # 2 次测量只能给「两次是否一致」；给不出「这一块到底该判什么」。
    # repeats≥3 时用票型分布看：全是 0/k 和 k/k 说明判定是二值的（无边界带），
    # 中间那些票型才是真正「落在线上」的块。
    hist = Counter(sum(1 for v in vs if v) for vs in verdict_of.values())
    unanimous = hist.get(0, 0) + hist.get(args.repeats, 0)
    majority_v = {k: majority(v) for k, v in verdict_of.items()}
    disagree_maj = sum(
        1 for k, v in verdict_of.items()
        for x in v if x != majority_v[k]
    )
    total_votes = sum(len(v) for v in verdict_of.values())

    emit("\n" + "=" * 78)
    emit(f"二、票型分布（每块判 {args.repeats} 次，横轴=判「相关」的次数）")
    emit("=" * 78)
    for k in range(args.repeats + 1):
        c = hist.get(k, 0)
        bar = "█" * int(round(c / max(n, 1) * 40))
        emit(f"  {k}/{args.repeats}: {c:>4}  {rate(c, n):.4f}  {bar}")
    emit(f"  全票一致（0 票或全票）: {unanimous}/{n} = {rate(unanimous, n):.4f}  "
         f"← 这些块判定是确定的")
    emit(f"  边界块（中间票型）    : {n - unanimous}/{n} = {rate(n - unanimous, n):.4f}")
    emit(f"  单次判定与多数票不符  : {disagree_maj}/{total_votes} = "
         f"{rate(disagree_maj, total_votes):.4f}  ← 单次测量的「翻错」概率")

    # ---------- 五、按题型 / 配置拆 ----------
    emit("\n" + "=" * 78)
    emit("三、分题型（同一对在多个题型下只属其一，按 qid 归属）")
    emit("=" * 78)
    seen_q: Dict[tuple, str] = {}
    for b in blocks:
        seen_q[(b["question"], b["text"])] = b["qtype"]
    by_type_stat: Dict[str, List[tuple]] = defaultdict(list)
    for key, vs in verdict_of.items():
        by_type_stat[seen_q[key]].append((vs[0], vs[1]))
    for t in sorted(by_type_stat):
        ps = by_type_stat[t]
        a = sum(1 for x, y in ps if x == y)
        emit(f"  {t:<14} n={len(ps):<4} 一致 {rate(a, len(ps)):.4f}  κ={cohens_kappa(ps):.4f}")

    # ---------- 六、对最终指标的影响（这才是决策相关的量）----------
    emit("\n" + "=" * 78)
    emit("四、判定抖动 → 指标抖动（Hit@1 / Hit@k：单次 vs 多数票）")
    emit("=" * 78)
    metric_rows: List[Dict[str, Any]] = []
    for cfg in args.configs:
        qids = sorted({b["qid"] for b in blocks if b["cfg"] == cfg})
        h1 = [0, 0]        # [单次(第1次), 多数票]
        hk = [0, 0]
        flip_q = []
        for qid in qids:
            bs = [b for b in blocks if b["cfg"] == cfg and b["qid"] == qid]
            fh_single = first_hit(bs, 0)
            fh_vote = first_hit_vote(bs)
            for i, fh in ((0, fh_single), (1, fh_vote)):
                if fh == 1:
                    h1[i] += 1
                if fh is not None:
                    hk[i] += 1
            if fh_single != fh_vote:
                flip_q.append(f"{qid}({fh_single}->{fh_vote})")
        metric_rows.append({
            "config": cfg,
            "n_questions": len(qids),
            "hit_at_1": h1,
            "hit_at_k": hk,
            "first_hit_changed": flip_q,
        })
        emit(f"  {cfg:<8} Hit@1 = {h1[0]}/{len(qids)} (单次) vs {h1[1]}/{len(qids)} (多数票)   "
             f"Hit@{args.top_k} = {hk[0]}/{len(qids)} vs {hk[1]}/{len(qids)}")
        emit(f"           首命中位置变化的题：{flip_q if flip_q else '无'}")

    # ---------- 七、翻转样例（供人工判断「抖得是否合理」）----------
    flips = [(k, v) for k, v in verdict_of.items() if v[0] != v[1]]
    emit("\n" + "=" * 78)
    emit(f"五、翻转样例（共 {len(flips)} 个，最多列 10 个）")
    emit("=" * 78)
    for key, vs in flips[:10]:
        b = uniq[key]
        emit(f"  [{b['qid']}] {vs[0]} vs {vs[1]}   {b['source']} p{b['page']}")
        emit(f"      理由A: {reason_of[key][0]}")
        emit(f"      理由B: {reason_of[key][1]}")
        emit(f"      块: {' '.join(b['text'].split())[:120]}…")

    emit("\n" + "=" * 78)
    emit(f"耗时 {elapsed:.0f}s，调用 {judge.calls} 次，失败 {judge.failures} 次")
    if judge.failures:
        emit("  ⚠ 有调用失败被降级成「不相关」—— 它会被误算成「判定不一致」，本次数据不可信")
    emit("=" * 78)

    # ---------- 八、落盘 ----------
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "config": {
                "questions_file": str(args.questions),
                "n_questions": len(picked),
                "qids": [q["id"] for q in picked],
                "repeats": args.repeats,
                "top_k": args.top_k,
                "configs": args.configs,
                "seed": args.seed,
                "judge": judge.model_id,
            },
            "summary": {
                "n_pairs": n,
                "agreement": rate(agree, n),
                "kappa": round(cohens_kappa(pairs), 4),
                "flip_false_to_true": f2t,
                "flip_true_to_false": t2f,
                "n_positive": len(pos),
                "positive_rate": rate(len(pos), n),
                "vote_hist": {str(k): v for k, v in sorted(hist.items())},
                "unanimous": unanimous,
                "unanimous_rate": rate(unanimous, n),
                "single_vs_majority_disagree": disagree_maj,
                "total_votes": total_votes,
                "calls": judge.calls,
                "failures": judge.failures,
                "elapsed_sec": round(elapsed, 1),
            },
            "metrics": metric_rows,
            "pairs": [
                {
                    "qid": b["qid"],
                    "qtype": b["qtype"],
                    "cfg": b["cfg"],
                    "rank": b["rank"],
                    "source": b["source"],
                    "page": b["page"],
                    "question": b["question"],
                    "text": b["text"][:400],
                    "verdicts": b["verdicts"],
                    "reasons": b["reasons"],
                }
                for b in blocks
            ],
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    emit(f"\n结果已保存 -> {out_path}")

    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"报告已写入 -> {args.out}")


if __name__ == "__main__":
    main()
