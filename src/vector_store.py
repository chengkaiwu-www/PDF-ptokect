"""
向量库：统一接口 + 两个后端。

  BruteForceVectorStore —— 手写 numpy 暴力检索，O(n·d)，**精确**（等价于真值）。
      保留它不是因为"没写好才留着"，而是：
        1) 它是 benchmark 里 Recall@k 的真值来源（近似索引准不准，得有个基准比）；
        2) 只有自己写过暴力检索，才真正理解 ANN 到底在解决什么问题。
  FaissVectorStore —— FAISS 索引，支持 flat / hnsw / ivf 三种类型与增量写入。

两个后端共享三条约定（S3 的核心改动）：

  1. **入库时 L2 归一化**。归一化后 余弦相似度 == 内积，检索省掉全库范数计算；
     且内积分数落在 [-1,1]，chunk 之间可比 —— 否则"阈值"这个概念根本不成立。
  2. **top-k 用 argpartition**，不做全排序（只要前 k 个却排全部是 O(n log n) 的浪费）。
  3. **"取几条"和"够不够格"是两层**。top-k 决定喂多少给 LLM，阈值决定这块合不合格。
     S3 之前 `if score > 0.1` 把两层压在一起：召回数会莫名少于 top_k，
     而且 0.1 这个数与模型分数分布强绑定（bge 和 MiniLM 尺度完全不同），换模型即废。
     现在阈值来自 config，默认不过滤。

存储布局（data/vector_store/）：
    documents.json   {meta, documents, doc_ids}   两个后端共用
    embeddings.npy   (n, d) float32，已归一化      两个后端共用
    faiss.index      FAISS 序列化索引              仅 faiss 后端

把 embeddings.npy 单独存下来的三个好处：
    · 换后端（brute <-> faiss）不必重新 embedding，直接从向量重建索引；
    · 换 FAISS 索引类型（hnsw <-> ivf）同理；
    · 离线 benchmark 可以直接读它，不依赖解析 PDF。
代价只是 1860 x 512 x 4B ≈ 3.8MB。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.config import (
    EMBEDDING_MODEL,
    FAISS_EF_CONSTRUCTION,
    FAISS_EF_SEARCH,
    FAISS_INDEX_TYPE,
    FAISS_M,
    FAISS_NLIST,
    FAISS_NPROBE,
    INGEST_NORMALIZE,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_TOP_K,
    VECTOR_BACKEND,
)
from src.vector_utils import content_hash, l2_normalize
from utils.logger import setup_logger

logger = setup_logger(__name__)

DOCUMENTS_FILE = "documents.json"
EMBEDDINGS_FILE = "embeddings.npy"
FAISS_INDEX_FILE = "faiss.index"
LEGACY_INDEX_FILE = "index.json"


class BaseVectorStore:
    """向量库公共逻辑：文档存储、去重、持久化、检索入口。"""

    backend_name = "base"

    def __init__(self, persist_directory: Path):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.documents: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []          # 与 documents 一一对应的内容 sha1
        self.meta: Dict[str, Any] = {}
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        self.documents_path = self.persist_directory / DOCUMENTS_FILE
        self.embeddings_path = self.persist_directory / EMBEDDINGS_FILE

    # ------------------------------------------------------------------ 写入

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        入库：归一化 -> 内容 hash 去重 -> 追加到向量与文档列表。

        Returns:
            实际新增的 chunk 数（重复的会被跳过）
        """
        if not documents:
            return 0
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents({len(documents)}) 与 embeddings({len(embeddings)}) 数量不一致"
            )

        vecs = np.asarray(embeddings, dtype=np.float32)
        if INGEST_NORMALIZE:
            vecs = l2_normalize(vecs)

        hashes = [content_hash(d.get("page_content", "")) for d in documents]

        known = set(self.doc_ids)
        keep: List[int] = []
        seen_in_batch = set()
        duplicated = 0
        for i, h in enumerate(hashes):
            if h in known or h in seen_in_batch:
                duplicated += 1
                continue
            seen_in_batch.add(h)
            keep.append(i)

        if duplicated:
            logger.info(f"去重：跳过 {duplicated} 个重复 chunk（内容 sha1 命中）")
        if not keep:
            logger.info("去重后没有新内容需要入库")
            return 0

        new_vecs = vecs[keep]
        self.vectors = new_vecs if self.vectors.size == 0 else np.vstack([self.vectors, new_vecs])
        self._index_add(new_vecs)

        for i in keep:
            self.documents.append(documents[i])
            self.doc_ids.append(hashes[i])

        self.meta.update(self._build_meta())
        logger.info(f"已入库 {len(keep)} 个 chunk（当前共 {len(self.documents)} 个）")
        return len(keep)

    def _index_add(self, vectors: np.ndarray) -> None:
        """后端钩子：把新向量加到索引里。暴力检索无需索引，默认空实现。"""

    # ------------------------------------------------------------------ 检索

    def similarity_search(
        self,
        query_embedding: List[float],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
        with_scores: bool = False,
    ) -> List[Any]:
        """
        检索 top-k。

        Args:
            query_embedding: 查询向量（未归一化也可以，这里会归一化）
            k: 返回条数，默认取 config.RETRIEVAL_TOP_K
            threshold: 相似度阈值；None 表示不过滤（或取 config 的默认值）。
                       想显式关闭过滤请传 float("-inf")。
            with_scores: True 时返回 [(document, score), ...]

        Returns:
            List[dict] 或 List[Tuple[dict, float]]
        """
        if len(self.documents) == 0:
            return []

        k = RETRIEVAL_TOP_K if k is None else k
        if threshold is None:
            threshold = RETRIEVAL_SCORE_THRESHOLD

        k = min(k, len(self.documents))
        indices, scores = self._search_topk(l2_normalize(query_embedding), k)

        pairs: List[Tuple[int, float]] = [
            (int(i), float(s)) for i, s in zip(indices, scores) if i >= 0
        ]
        # 阈值是独立的一层：先取满 top-k，再按阈值筛掉不合格的
        dropped = 0
        if threshold is not None:
            before = len(pairs)
            pairs = [(i, s) for i, s in pairs if s >= threshold]
            dropped = before - len(pairs)

        logger.info(
            f"Found {len(pairs)} similar document(s)"
            + (f"（阈值 {threshold} 过滤掉 {dropped} 条）" if dropped else "")
        )
        if with_scores:
            return [(self.documents[i], s) for i, s in pairs]
        return [self.documents[i] for i, s in pairs]

    def _search_topk(self, query_vec: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        raise NotImplementedError

    # -------------------------------------------------------------- 持久化

    def _build_meta(self) -> Dict[str, Any]:
        dim = int(self.vectors.shape[1]) if self.vectors.size else 0
        meta = {
            "backend": self.backend_name,
            "count": len(self.documents),
            "dim": dim,
            "normalized": INGEST_NORMALIZE,
            "embedding_model": str(EMBEDDING_MODEL),
        }
        # 增量索引的账本：{文件名: {sha1, chunks}}，由 main.sync_vector_store 维护。
        # 必须在这里透传，否则 save() 里的 meta 重建会把它丢掉。
        if "indexed_files" in self.meta:
            meta["indexed_files"] = self.meta["indexed_files"]
        meta.update(self._extra_meta())
        return meta

    def _extra_meta(self) -> Dict[str, Any]:
        return {}

    def _save_vectors(self) -> None:
        np.save(self.embeddings_path, self.vectors)

    def _load_vectors(self) -> bool:
        if not self.embeddings_path.exists():
            return False
        self.vectors = np.load(self.embeddings_path).astype(np.float32)
        return True

    def save(self) -> None:
        self._save_vectors()
        payload = {
            "meta": self._build_meta(),
            "documents": self.documents,
            "doc_ids": self.doc_ids,
        }
        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        self.meta = payload["meta"]
        logger.info(f"向量库已保存到 {self.persist_directory}（{len(self.documents)} chunk）")

    def load(self) -> bool:
        if not self.documents_path.exists():
            return False
        try:
            with open(self.documents_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.documents = payload.get("documents", [])
            self.doc_ids = payload.get("doc_ids", [])
            self.meta = payload.get("meta", {})
            if not self._load_vectors():
                logger.error(f"未找到 {EMBEDDINGS_FILE}，向量库不完整")
                return False
            logger.info(
                f"已加载向量库：{len(self.documents)} chunk，"
                f"维度 {self.meta.get('dim')}，后端 {self.meta.get('backend')}"
            )
            return True
        except Exception as e:
            logger.error(f"加载向量库失败: {e}")
            return False

    def __len__(self) -> int:
        return len(self.documents)

    def needs_persist(self) -> bool:
        """
        磁盘上的产物是否与当前配置不一致（需要重新落盘）。

        典型场景：上次用 brute 建的库，这次 VECTOR_BACKEND=faiss —— 内存里能从
        embeddings.npy 重建出 FAISS 索引，但磁盘上还没有 faiss.index，
        不落盘的话每次启动都要白重建一遍。
        """
        return (self.meta or {}).get("backend") != self.backend_name


class BruteForceVectorStore(BaseVectorStore):
    """
    手写暴力余弦检索（numpy 实现），O(n·d)，结果精确。

    检索一行就够：`scores = V @ q`。之所以留着它，是因为它是
    「ANN 牺牲了多少召回率」这个问题的唯一答案来源（真值）。
    """

    backend_name = "brute"

    def _search_topk(self, query_vec: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        # 向量在入库时已归一化，查询向量在上层也归一化了 -> 内积即余弦
        scores = self.vectors @ query_vec
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            # 只要 top-k 却做全排序是浪费；argpartition 是 O(n)
            part = np.argpartition(-scores, k - 1)[:k]
            order = part[np.argsort(-scores[part])]
        return order.tolist(), scores[order].tolist()


class FaissVectorStore(BaseVectorStore):
    """
    FAISS 索引后端。三种类型对应三种思路：

      flat —— IndexFlatIP，暴力内积，精确，O(n)。作为 FAISS 侧的对照。
      hnsw —— IndexHNSWFlat，分层可导航小世界图。近似检索，查询近似 O(log n)。
              关键参数：M（连接数）、efConstruction（建索引候选池）、
                        efSearch（查询候选池，**可在运行时调**，用来换"准/快"）。
      ivf  —— IndexIVFFlat，倒排文件。先用 k-means 聚成 nlist 个簇，
              查询时只搜最近的 nprobe 个簇。需要先 train。

    无论哪种，索引都是建在**归一化后的向量**上的，所以用内积度量（IP）即等价余弦。
    """

    backend_name = "faiss"

    def __init__(self, persist_directory: Path, index_type: Optional[str] = None):
        super().__init__(persist_directory)
        self.index_type = (index_type or FAISS_INDEX_TYPE).lower()
        self.index = None
        self.index_path = self.persist_directory / FAISS_INDEX_FILE

    def _extra_meta(self) -> Dict[str, Any]:
        return {
            "index_type": self.index_type,
            "faiss_metric": "inner_product",
            "hnsw": {"M": FAISS_M, "efConstruction": FAISS_EF_CONSTRUCTION,
                     "efSearch": FAISS_EF_SEARCH},
            "ivf": {"nlist": FAISS_NLIST, "nprobe": FAISS_NPROBE},
        }

    def needs_persist(self) -> bool:
        meta = self.meta or {}
        if meta.get("backend") != self.backend_name:
            return True
        if meta.get("index_type") != self.index_type:
            return True
        return not self.index_path.exists()

    def _new_index(self, dim: int):
        import faiss

        if self.index_type == "flat":
            return faiss.IndexFlatIP(dim)

        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, FAISS_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
            index.hnsw.efSearch = FAISS_EF_SEARCH
            return index

        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, FAISS_NLIST, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = FAISS_NPROBE
            return index

        raise ValueError(
            f"未知 FAISS_INDEX_TYPE: {self.index_type}（可选 flat / hnsw / ivf）"
        )

    def _index_add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.index is None:
            self.index = self._new_index(vectors.shape[1])
            logger.info(f"创建 FAISS 索引：type={self.index_type} dim={vectors.shape[1]}")

        # IVF 必须先训练（k-means 找簇心）。训练数据少于 nlist*39 时 FAISS 会警告，
        # 但不会失败——这里显式提示，避免用户误以为索引质量很好。
        if not self.index.is_trained:
            n_train = vectors.shape[0]
            if self.index_type == "ivf" and n_train < FAISS_NLIST * 39:
                logger.warning(
                    f"IVF 训练样本偏少（{n_train} < nlist*39 = {FAISS_NLIST * 39}），"
                    f"簇心质量可能不佳；语料变大后建议重建索引"
                )
            logger.info(f"训练 FAISS 索引（{n_train} 个向量，nlist={FAISS_NLIST}）")
            self.index.train(vectors)

        self.index.add(vectors)

    def _search_topk(self, query_vec: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        if self.index is None or self.index.ntotal == 0:
            return [], []
        if self.index_type == "hnsw":
            # efSearch 是运行时参数：调大 -> 召回更准、延迟更高。
            # 同一个索引可以按场景动态换策略，这是 HNSW 最实用的地方。
            self.index.hnsw.efSearch = FAISS_EF_SEARCH
        elif self.index_type == "ivf":
            self.index.nprobe = FAISS_NPROBE

        scores, indices = self.index.search(query_vec.reshape(1, -1), k)
        return indices[0].tolist(), scores[0].tolist()

    def _save_vectors(self) -> None:
        super()._save_vectors()
        if self.index is not None:
            import faiss

            # 不用 faiss.write_index：它内部是 C++ 的 fopen，在 Windows 上遇到
            # 非 ASCII 路径（例如 "D:\代码\PDF-ptokect\..."）会直接失败：
            #   Error: 'f' failed: could not open ... for writing: No such file or directory
            # serialize_index 只做内存序列化，落盘交给 Python 的 open()，
            # 由 Python 处理 Unicode 路径 —— 彻底绕开这个问题。
            blob = faiss.serialize_index(self.index)
            self.index_path.write_bytes(blob.tobytes())

    def _load_vectors(self) -> bool:
        if not super()._load_vectors():
            return False
        if self.vectors.size == 0:
            return True

        saved_type = (self.meta or {}).get("index_type")
        need_rebuild = False

        if self.index_path.exists() and saved_type == self.index_type:
            import faiss

            blob = np.frombuffer(self.index_path.read_bytes(), dtype=np.uint8).copy()
            self.index = faiss.deserialize_index(blob)
            if self.index.d != self.vectors.shape[1]:
                logger.warning("FAISS 索引维度与向量不一致，改为重建")
                need_rebuild = True
        else:
            if self.index_path.exists():
                logger.info(f"索引类型从 {saved_type} 变为 {self.index_type}，重建索引")
            else:
                logger.info("未找到 FAISS 索引文件，从 embeddings.npy 重建")
            need_rebuild = True

        if need_rebuild or self.index is None or self.index.ntotal != self.vectors.shape[0]:
            if not need_rebuild:
                logger.warning("FAISS 索引条数与向量不一致，改为重建")
            self.index = None
            self._index_add(self.vectors)
        return True


# --------------------------------------------------------------------- 工厂

_BACKENDS = {
    "brute": BruteForceVectorStore,
    "faiss": FaissVectorStore,
}


def _resolve_backend(backend: Optional[str]):
    name = (backend or VECTOR_BACKEND).lower()
    if name not in _BACKENDS:
        raise ValueError(f"未知 VECTOR_BACKEND: {name}（可选 {' / '.join(_BACKENDS)}）")
    return name, _BACKENDS[name]


def _instantiate(cls, persist_directory: Path, index_type: Optional[str] = None):
    """按后端构造实例；index_type 只对 FAISS 后端有意义。"""
    if cls is FaissVectorStore:
        return cls(persist_directory, index_type=index_type)
    return cls(persist_directory)


def vector_store_exists(persist_directory: Path, backend: Optional[str] = None) -> bool:
    """
    判断持久化的向量库是否存在且完整。

    不同后端的判据不同：FAISS 后端需要 faiss.index（缺失时虽可从
    embeddings.npy 重建，但算"未构建完成"更符合直觉）。
    """
    _, cls = _resolve_backend(backend)
    directory = Path(persist_directory)
    if not (directory / DOCUMENTS_FILE).exists():
        return False
    if not (directory / EMBEDDINGS_FILE).exists():
        return False
    if cls is FaissVectorStore and not (directory / FAISS_INDEX_FILE).exists():
        return False
    return True


def _legacy_migrate(persist_directory: Path):
    """
    把 S3 之前的 index.json 迁移成新格式。

    旧格式（S3 之前）：
        {"embeddings": [[...]], "documents": [{...}]}   —— 未归一化，无 doc_ids
    新格式拆成 documents.json + embeddings.npy，并在入库路径上做 L2 归一化。
    这里读旧文件后直接交给 add_documents 走标准流程（含归一化与补算 doc_ids）。
    """
    legacy = Path(persist_directory) / LEGACY_INDEX_FILE
    if not legacy.exists():
        return None
    try:
        with open(legacy, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents = data.get("documents", [])
        embeddings = data.get("embeddings", [])
        if not documents or not embeddings:
            return None
        logger.info(f"发现旧版索引 {legacy.name}，正在迁移到新格式（会补 L2 归一化与内容指纹）")
        return documents, embeddings
    except Exception as e:
        logger.error(f"读取旧版索引失败: {e}")
        return None


def create_vector_store(
    documents: List[Dict[str, Any]],
    embeddings: List[List[float]],
    persist_directory: Path,
    backend: Optional[str] = None,
    index_type: Optional[str] = None,
):
    """新建向量库并持久化。"""
    _, cls = _resolve_backend(backend)
    store = _instantiate(cls, persist_directory, index_type)
    store.add_documents(documents, embeddings)
    store.save()
    return store


def load_vector_store(
    persist_directory: Path,
    backend: Optional[str] = None,
    index_type: Optional[str] = None,
):
    """
    加载已有向量库；若只有 S3 之前的 index.json，则自动迁移。

    Returns:
        向量库实例；不存在或加载失败时返回 None
    """
    _, cls = _resolve_backend(backend)
    store = _instantiate(cls, persist_directory, index_type)
    if store.load():
        return store

    migrated = _legacy_migrate(persist_directory)
    if migrated is None:
        return None

    documents, embeddings = migrated
    store = _instantiate(cls, persist_directory, index_type)
    store.add_documents(documents, embeddings)
    store.save()
    logger.info(f"迁移完成（{len(store)} chunk）；旧 {LEGACY_INDEX_FILE} 已保留但不再使用")
    return store


def create_retriever(vector_store: BaseVectorStore, k: int = RETRIEVAL_TOP_K):
    """
    生成检索函数（保持原有闭包签名，便于 RAGPipeline 注入）。

    Args:
        vector_store: 向量库实例（brute 或 faiss 均可）
        k: 召回条数
    """

    def retriever(query: str, embedding_model) -> List[Dict[str, Any]]:
        query_embedding = embedding_model.embed_query(query)
        return vector_store.similarity_search(query_embedding, k=k)

    return retriever
