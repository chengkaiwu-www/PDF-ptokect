# 必须先导入 config：它会在模块顶部设置 HF_ENDPOINT（镜像站），
# 早于 sentence_transformers / huggingface_hub 的导入才能生效。
from config.config import EMBEDDING_MODEL

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from utils.logger import setup_logger

logger = setup_logger(__name__)


class LocalEmbeddings(Embeddings):
    """
    基于 sentence-transformers 的本地 embedding 模型。

    默认加载 config.EMBEDDING_MODEL 指定的模型，也可由构造参数覆盖。
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding model loaded (dim={self.model.get_embedding_dimension()})")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


def get_embedding_model() -> Embeddings:
    """
    Get the embedding model based on configuration.

    Returns:
        Embeddings instance
    """
    logger.info("Using local sentence-transformers embedding model")
    return LocalEmbeddings()


def create_embeddings(texts: List[str], embedding_model: Embeddings) -> List[List[float]]:
    """
    Create embeddings for the given texts.

    Args:
        texts: List of text strings
        embedding_model: Embeddings model instance

    Returns:
        List of embedding vectors
    """
    if not texts:
        logger.warning("No texts to embed")
        return []

    logger.info(f"Creating embeddings for {len(texts)} text(s)")
    embeddings = embedding_model.embed_documents(texts)
    logger.info(f"Created embeddings with dimension: {len(embeddings[0])}")

    return embeddings
