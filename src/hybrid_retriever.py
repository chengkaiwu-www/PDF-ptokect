"""
混合检索：稠密（向量）+ 稀疏（BM25），用 RRF 融合，可选 Cross-Encoder 精排。

为什么需要混合 —— 两种检索的**错误模式不同**，这才是融合的意义：

  向量（稠密）          BM25（稀疏）
  ─────────────────     ─────────────────
  强：同义改写、语义相近   强：专有名词、数字、编号、代码标识符
  弱：罕见词被"平均"掉     弱：换个说法就完全召回不到（零字面重叠 = 0 分）

  一个只靠语义、一个只靠字面，两者的失败集合的**交集不大**，
  所以取并集再重排能吃到两边的长处。

融合为什么用 RRF 而不是"加权分数和"：
  余弦分数 ∈ [-1, 1]（归一化后），BM25 分数无上界（随语料、k1/b 变化），
  直接相加或加权相加会让量纲大的一方完全支配结果，权重也失去意义。
  RRF 只看**名次**，对分数尺度免疫，因此跨模型/跨语料不需要重新调参：

      RRF(d) = Σ_{r ∈ 各召回通道} 1 / (k + rank_r(d))     k 常取 60

  代价：丢掉了"分数差距"信息。第 1 名和第 2 名分数差 0.3 还是 0.001，
  RRF 眼里都一样 —— 所以它适合做**融合（求稳）**，不适合做**精排（求准）**。
  精排交给 Cross-Encoder。

三段式流水线（工业界标准形状，本项目按这个顺序落地）：
    ① 召回：向量 top-50 + BM25 top-50      —— 宁多勿漏，两条通道
    ② 融合：RRF 合成一个 top-N              —— 只看名次，稳
    ③ 精排：Cross-Encoder 给 top-N 打分      —— 宁准勿滥，慢但准
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.bm25 import BM25Index
from utils.logger import setup_logger

logger = setup_logger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[int, float]]:
    """
    RRF 融合：输入若干个「按相关度降序排列的 doc 下标列表」，输出融合后的排名。

    Args:
        ranked_lists: 每个通道的召回结果（已按降序排好），元素是 doc 下标
        k: 平滑常数。作用有两个：
           (1) 防止 rank=0 时除零（本项目 rank 从 1 开始，但公式上仍需要）；
           (2) 控制「头部优势」的强度 —— k 越小，第 1 名越碾压后面；
               k 越大，各名次越接近。60 是行业经验值（源自 Cormack 2009 的论文）。
        weights: 每个通道的权重，默认全 1

    Returns:
        [(doc_idx, rrf_score), ...] 按 rrf_score 降序

    直观数值（k=60）：
        rank 1 -> 1/61 = 0.01639
        rank 2 -> 1/62 = 0.01613
        rank 5 -> 1/65 = 0.01538
    注意这个分数与「相似度」完全不是一个量纲，只能用来排序，不能当阈值。
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: Dict[int, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc_idx in enumerate(ranked, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """
    向量 + BM25 混合检索器，可挂 Cross-Encoder 精排。

    对外签名兼容 vector_store.create_retriever 返回的闭包
    （`retriever(query, embedding_model)`），并额外提供 retrieve() 以支持
    candidate_k / top_k / 中间诊断信息。
    """

    def __init__(
        self,
        vector_store,
        *,
        use_bm25: bool = True,
        use_vector: bool = True,
        rrf_k: int = 60,
        candidate_k: int = 50,
        reranker=None,
        top_k: int = 4,
        channel_weights: Optional[Dict[str, float]] = None,
    ):
        self.vector_store = vector_store
        self.use_bm25 = use_bm25
        self.use_vector = use_vector
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.reranker = reranker
        self.top_k = top_k
        # 各通道融合权重，默认等权。等权是 RRF 的卖点（无需调参），
        # 但当两个通道质量差距明显时，等权会把强通道拖下来 ——
        # 这时加权 RRF 是补救手段，权重本身必须用评测集标定。
        self.channel_weights = channel_weights or {}

        self.bm25: Optional[BM25Index] = None
        if use_bm25:
            self._ensure_bm25()

    # ------------------------------------------------------------------ 检索

    def _ensure_bm25(self) -> BM25Index:
        """懒构建 BM25 索引（语料规模变化时自动重建）。"""
        if self.bm25 is None or self.bm25.n_docs != len(self.vector_store):
            self.bm25 = BM25Index(self.vector_store.documents)
        return self.bm25

    def _vector_recall(self, query: str, embedding_model, k: int) -> List[Tuple[int, float]]:
        """向量通道：返回 [(doc 下标, 余弦分数)]，降序。阈值显式关闭，保证每通道都取满。"""
        query_vec = embedding_model.embed_query(query)
        hits = self.vector_store.similarity_search(
            query_vec, k=k, threshold=float("-inf"), with_scores=True
        )
        # similarity_search 返回的是 document dict，需要反查下标。
        # 入库已按内容 sha1 去重，这里用对象身份映射最直接。
        id_to_idx = {id(d): i for i, d in enumerate(self.vector_store.documents)}
        return [
            (id_to_idx[id(d)], float(score))
            for d, score in hits
            if id(d) in id_to_idx
        ]

    def _bm25_recall(self, query: str, k: int) -> List[Tuple[int, float]]:
        index = self._ensure_bm25()
        return [(doc_idx, float(score)) for doc_idx, score in index.search(query, k=k)]

    def retrieve(
        self,
        query: str,
        embedding_model,
        top_k: Optional[int] = None,
        return_debug: bool = False,
    ) -> Any:
        """
        完整检索流程：多通道召回 -> RRF 融合 ->（可选）Cross-Encoder 精排 -> 取 top_k。

        Returns:
            默认返回 List[dict]；return_debug=True 时返回
            {"documents": [...], "debug": {...}}，含每通道名次与分数，
            便于在学习/调参时看清「谁把这块捞上来的」。
        """
        top_k = self.top_k if top_k is None else top_k
        n_docs = len(self.vector_store.documents)
        if n_docs == 0:
            return {"documents": [], "debug": {}} if return_debug else []

        candidate_k = max(self.candidate_k, top_k)

        channels: List[Tuple[str, List[Tuple[int, float]]]] = []
        if self.use_vector:
            channels.append(("vector", self._vector_recall(query, embedding_model, candidate_k)))
        if self.use_bm25:
            channels.append(("bm25", self._bm25_recall(query, candidate_k)))

        if not channels:
            raise ValueError("至少需要启用一个召回通道（use_vector / use_bm25）")

        # 单通道时不走 RRF：RRF 会丢掉分数，而单通道的原始分数是有意义的
        # （余弦相似度可比、BM25 可看相对大小）。这是「不要为了统一而统一」的例子。
        if len(channels) == 1:
            name, scored = channels[0]
            ordered = list(scored)
            fused_scores: Dict[int, float] = dict(scored)
            score_kind = "cosine" if name == "vector" else "bm25"
        else:
            ordered = reciprocal_rank_fusion(
                [[doc_idx for doc_idx, _ in ranked] for _, ranked in channels],
                k=self.rrf_k,
                weights=[self.channel_weights.get(ch, 1.0) for ch, _ in channels],
            )
            fused_scores = dict(ordered)
            score_kind = "rrf"
            name = "hybrid"

        docs = self.vector_store.documents

        # ---- 精排：只对融合后的前 candidate_k 个候选做，不是全库
        rerank_scores: Dict[int, float] = {}
        if self.reranker is not None and ordered:
            pool = [doc_idx for doc_idx, _ in ordered[:candidate_k]]
            texts = [docs[i].get("page_content", "") for i in pool]
            scores = self.reranker.score(query, texts)
            rerank_scores = dict(zip(pool, scores))
            ordered = sorted(zip(pool, scores), key=lambda kv: kv[1], reverse=True)

        selected = ordered[:top_k]

        if return_debug:
            rank_maps = {
                ch_name: {doc_idx: r for r, (doc_idx, _) in enumerate(ranked, start=1)}
                for ch_name, ranked in channels
            }
            debug = {
                "channels": {ch: len(ranked) for ch, ranked in channels},
                "score_kind": "rerank" if rerank_scores else score_kind,
                "rows": [
                    {
                        "doc_idx": doc_idx,
                        "ranks": {ch: rank_maps[ch].get(doc_idx) for ch in rank_maps},
                        "fused": round(fused_scores.get(doc_idx, 0.0), 6),
                        "rerank": round(rerank_scores[doc_idx], 4)
                        if doc_idx in rerank_scores else None,
                    }
                    for doc_idx, _ in selected
                ],
            }
            return {"documents": [docs[i] for i, _ in selected], "debug": debug, "mode": name}

        return [docs[i] for i, _ in selected]

    # 兼容 create_retriever 的闭包签名
    def __call__(self, query: str, embedding_model, top_k: Optional[int] = None):
        return self.retrieve(query, embedding_model, top_k=top_k)


def create_hybrid_retriever(
    vector_store,
    mode: Optional[str] = None,
    *,
    candidate_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    rerank: Optional[bool] = None,
    reranker_model: Optional[str] = None,
    top_k: Optional[int] = None,
    channel_weights: Optional[Dict[str, float]] = None,
) -> HybridRetriever:
    """
    统一工厂：main.py 与 eval/run_eval.py 都走这里。

    为什么要一个统一工厂：检索器有两种召回通道、一个融合参数、一个可选精排，
    组装逻辑一旦在 CLI 和评测脚本里各写一份，就会「评测的配置和线上跑的不一样」
    —— 这是 RAG 项目最常见、也最难发现的评测失真。
    """
    from config.config import (
        RERANKER_MODEL,
        RERANK_ENABLED,
        RETRIEVAL_CANDIDATE_K,
        RETRIEVAL_MODE,
        RETRIEVAL_TOP_K,
        RRF_K,
    )

    mode = (mode or RETRIEVAL_MODE).lower()
    if mode not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"未知检索模式: {mode}（可选 vector / bm25 / hybrid）")

    candidate_k = RETRIEVAL_CANDIDATE_K if candidate_k is None else candidate_k
    rrf_k = RRF_K if rrf_k is None else rrf_k
    top_k = RETRIEVAL_TOP_K if top_k is None else top_k
    rerank = RERANK_ENABLED if rerank is None else rerank

    reranker = None
    if rerank:
        from src.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(reranker_model or RERANKER_MODEL)

    return HybridRetriever(
        vector_store,
        use_vector=(mode in ("vector", "hybrid")),
        use_bm25=(mode in ("bm25", "hybrid")),
        rrf_k=rrf_k,
        candidate_k=candidate_k,
        reranker=reranker,
        top_k=top_k,
        channel_weights=channel_weights,
    )
