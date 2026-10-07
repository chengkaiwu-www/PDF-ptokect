from typing import List
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LocalEmbeddings(Embeddings):
    """Local sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
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
