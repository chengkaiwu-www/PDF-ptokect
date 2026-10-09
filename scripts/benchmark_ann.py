"""
ANN 索引规模实验：暴力检索 vs FAISS(flat / hnsw / ivf)。

为什么需要这个脚本
------------------
当前语料只有约 1900 个 chunk，暴力检索一次查询在毫秒级就结束了——**在这个规模上
ANN 毫无优势可言**。要说明「ANN 到底解决了什么问题」，必须把规模放大到
10^5 量级再测。所以本脚本做的是：

    1. 取真实 embedding 的**维度**（512），按需生成 / 复制出不同规模的向量集；
    2. 用同一批查询，分别测 暴力 / FAISS-flat / FAISS-hnsw / FAISS-ivf 的
       建索引耗时 与 单次查询延迟；
    3. 以**暴力检索的结果为真值**，计算各近似索引的 Recall@k
       —— 这就是「用召回率换速度」里那个召回率。

两种造数模式
    synth  —— 归一化的高斯随机向量。ANN 领域的标准微基准，能干净地反映
               索引本身的行为，但不反映真实文本向量的分布。
    corpus —— 把真实语料向量复制到目标规模，并加一点高斯噪声
               （模拟「大语料里有大量相似但不相同的段落」）。
               比 synth 更接近真实分布，但相似度整体偏高。

结论只做横向比较（同一模式下不同索引之间），不要当成线上绝对性能。

用法：
  D:\\anaconda\\python.exe scripts/benchmark_ann.py
  D:\\anaconda\\python.exe scripts/benchmark_ann.py --mode corpus --sizes 2000,20000,100000
  D:\\anaconda\\python.exe scripts/benchmark_ann.py --real-queries        # 用 eval 里的真实问题当查询
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (  # noqa: E402
    VECTOR_STORE_DIR,
    FAISS_EF_CONSTRUCTION,
    FAISS_EF_SEARCH,
    FAISS_M,
    FAISS_NLIST,
    FAISS_NPROBE,
)
from src.vector_utils import l2_normalize  # noqa: E402

EF_SEARCH_GRID = [16, 64, 256]
NPROBE_GRID = [1, 8, 32]


# ------------------------------------------------------------------ 工具

def load_store_vectors(store_dir: Path) -> np.ndarray:
    path = store_dir / "embeddings.npy"
    if not path.exists():
        print(f"[ERROR] 未找到 {path}")
        print("        先执行: python main.py --recreate")
        sys.exit(1)
    return np.load(path).astype(np.float32)


def build_dataset(vecs: np.ndarray, n: int, queries: np.ndarray, mode: str, seed: int = 0):
    """
    造出规模为 n 的向量集与查询集。

    Returns: (xb, xq) 都已 L2 归一化
    """
    rng = np.random.default_rng(seed)
    dim = vecs.shape[1]
    n_q = queries.shape[0]

    if mode == "synth":
        xb = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
        xq = l2_normalize(rng.standard_normal((n_q, dim)).astype(np.float32))
        return xb, xq

    # corpus 模式：复制真实向量 + 小幅噪声，保持真实分布的同时制造"大量相似样本"
    reps = int(np.ceil(n / vecs.shape[0]))
    xb = np.tile(vecs, (reps, 1))[:n]
    xb = xb + rng.normal(0, 0.05, xb.shape).astype(np.float32)
    xq = queries + rng.normal(0, 0.05, queries.shape).astype(np.float32)
    return l2_normalize(xb), l2_normalize(xq)


def brute_topk(xb: np.ndarray, xq: np.ndarray, k: int) -> np.ndarray:
    """暴力内积检索，返回 (n_q, k) 的下标。作为 Recall 的真值。"""
    scores = xq @ xb.T
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)[:, :k]
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return part[rows, order]


def recall_at_k(truth: np.ndarray, got: np.ndarray, k: int) -> float:
    """以暴力检索结果为真值的 Recall@k（两个列表的平均重合比例）。"""
    hits = 0
    total = 0
    for t, g in zip(truth, got):
        t_set = set(t[:k].tolist())
        g_set = set(x for x in g[:k].tolist() if x >= 0)
        hits += len(t_set & g_set)
        total += len(t_set)
    return hits / total if total else 0.0


def timed(fn, repeats: int = 1) -> Tuple[float, object]:
    """跑 repeats 次取平均耗时（毫秒），返回 (ms, 最后一次结果)。"""
    fn()  # 预热，排除首次分配/惰性初始化的影响
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    return (time.perf_counter() - t0) * 1000 / repeats, out


# ------------------------------------------------------------------ 单规模测试

def run_one_size(xb: np.ndarray, xq: np.ndarray, k: int, repeats: int) -> List[Dict]:
    import faiss

    n, dim = xb.shape
    rows: List[Dict] = []

    truth = None
    ms, truth = timed(lambda: brute_topk(xb, xq, k), repeats=repeats)
    rows.append({
        "method": "brute (numpy)",
        "param": "-",
        "build_sec": 0.0,
        "latency_ms_per_query": ms / xq.shape[0],
        "recall_at_k": 1.0,
    })

    # ---- FAISS flat：精确，作为"FAISS 侧"的对照 ----
    flat = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    flat.add(xb)
    build = time.perf_counter() - t0
    ms, got = timed(lambda: flat.search(xq, k)[1], repeats=repeats)
    rows.append({
        "method": "faiss flat",
        "param": "-",
        "build_sec": build,
        "latency_ms_per_query": ms / xq.shape[0],
        "recall_at_k": recall_at_k(truth, got, k),
    })

    # ---- FAISS HNSW：efSearch 决定"准/快" ----
    hnsw = faiss.IndexHNSWFlat(dim, FAISS_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
    t0 = time.perf_counter()
    hnsw.add(xb)
    build = time.perf_counter() - t0
    for ef in EF_SEARCH_GRID:
        hnsw.hnsw.efSearch = ef

        def search():
            hnsw.hnsw.efSearch = ef
            return hnsw.search(xq, k)[1]

        ms, got = timed(search, repeats=repeats)
        rows.append({
            "method": "faiss hnsw",
            "param": f"M={FAISS_M} efSearch={ef}",
            "build_sec": build,
            "latency_ms_per_query": ms / xq.shape[0],
            "recall_at_k": recall_at_k(truth, got, k),
        })

    # ---- FAISS IVF：nprobe 决定"准/快" ----
    nlist = min(FAISS_NLIST, max(4, n // 40))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivf.train(xb)
    ivf.add(xb)
    build = time.perf_counter() - t0
    for nprobe in NPROBE_GRID:
        if nprobe > nlist:
            continue

        def search():
            ivf.nprobe = nprobe
            return ivf.search(xq, k)[1]

        ms, got = timed(search, repeats=repeats)
        rows.append({
            "method": "faiss ivf",
            "param": f"nlist={nlist} nprobe={nprobe}",
            "build_sec": build,
            "latency_ms_per_query": ms / xq.shape[0],
            "recall_at_k": recall_at_k(truth, got, k),
        })

    return rows


# ------------------------------------------------------------------ 主流程

def print_table(all_rows: Dict[int, List[Dict]]) -> None:
    header = f"{'规模':>8}  {'方法':<14} {'参数':<26} {'建索引(s)':>10} {'延迟(ms/query)':>15} {'Recall@k':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for n, rows in all_rows.items():
        for i, r in enumerate(rows):
            label = f"{n:,}" if i == 0 else ""
            print(
                f"{label:>8}  {r['method']:<14} {r['param']:<26} "
                f"{r['build_sec']:>10.3f} {r['latency_ms_per_query']:>15.3f} "
                f"{r['recall_at_k']:>9.4f}"
            )
        print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="ANN 索引规模实验")
    parser.add_argument("--store-dir", type=str, default=str(VECTOR_STORE_DIR))
    parser.add_argument("--mode", choices=["synth", "corpus"], default="synth")
    parser.add_argument("--sizes", type=str, default="2000,20000,100000,200000")
    parser.add_argument("--queries", type=int, default=20, help="查询条数")
    parser.add_argument("--k", type=int, default=4, help="top-k")
    parser.add_argument("--repeats", type=int, default=3, help="每个配置重复次数（取平均）")
    parser.add_argument("--real-queries", action="store_true",
                        help="用 eval/questions.json 里的真实问题当查询（需加载 embedding 模型）")
    parser.add_argument("--out", type=str, default="", help="把结果保存为 JSON")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    store_dir = Path(args.store_dir)
    corpus = load_store_vectors(store_dir)
    dim = corpus.shape[1]

    # 查询向量
    rng = np.random.default_rng(42)
    if args.real_queries:
        from src.embedding import get_embedding_model

        print("加载 embedding 模型以生成真实问题向量 …")
        model = get_embedding_model()
        questions_path = PROJECT_ROOT / "eval" / "questions.json"
        data = json.loads(questions_path.read_text(encoding="utf-8"))
        questions = data["questions"] if isinstance(data, dict) else data
        texts = [q["question"] for q in questions][: args.queries]
        queries = l2_normalize(np.asarray(model.embed_documents(texts), dtype=np.float32))
        print(f"使用 {len(texts)} 个真实问题作为查询")
    else:
        idx = rng.choice(corpus.shape[0], size=min(args.queries, corpus.shape[0]), replace=False)
        queries = l2_normalize(corpus[idx])

    print(
        f"模式={args.mode}  维度={dim}  语料真实规模={corpus.shape[0]}  "
        f"查询数={queries.shape[0]}  top-k={args.k}  重复={args.repeats}"
    )

    all_rows: Dict[int, List[Dict]] = {}
    for n in sizes:
        xb, xq = build_dataset(corpus, n, queries, args.mode)
        print(f"\n>>> 规模 {n:,}（{args.mode}）…")
        all_rows[n] = run_one_size(xb, xq, args.k, args.repeats)
        for r in all_rows[n]:
            print(
                f"    {r['method']:<14} {r['param']:<26} "
                f"{r['latency_ms_per_query']:>8.3f} ms/q  recall={r['recall_at_k']:.4f}"
            )
        del xb
        gc.collect()

    print_table(all_rows)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": args.mode,
            "dim": dim,
            "corpus_size": int(corpus.shape[0]),
            "queries": int(queries.shape[0]),
            "k": args.k,
            "repeats": args.repeats,
            "results": {str(n): rows for n, rows in all_rows.items()},
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
