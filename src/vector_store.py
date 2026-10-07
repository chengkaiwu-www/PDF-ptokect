import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from config.config import RETRIEVAL_TOP_K
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SimpleVectorStore:
    """
    A simple vector store implementation that stores embeddings in memory and persists to disk.
    """

    def __init__(self, persist_directory: Path):
        self.persist_directory = persist_directory
        self.embeddings: List[List[float]] = []
        self.documents: List[Dict[str, Any]] = []
        self.index_path = persist_directory / "index.json"

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the store.

        Args:
            documents: List of document dicts with 'page_content' and 'metadata'
            embeddings: List of embedding vectors
        """
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        logger.info(f"Added {len(documents)} documents to vector store")

    def save(self):
        """Save the vector store to disk."""
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents
        }
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        logger.info(f"Vector store saved to {self.index_path}")

    def load(self) -> bool:
        """Load the vector store from disk."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.embeddings = data.get("embeddings", [])
            self.documents = data.get("documents", [])
            logger.info(f"Loaded {len(self.documents)} documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """
        Find the most similar documents to the query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of document dicts sorted by similarity
        """
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(self.embeddings)

        similarities = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )

        top_indices = np.argsort(similarities)[::-1][:k]
        results = [self.documents[i] for i in top_indices if similarities[i] > 0.1]

        logger.info(f"Found {len(results)} similar documents")
        return results

    def __len__(self):
        return len(self.documents)


def create_vector_store(documents: List[Dict[str, Any]], embeddings: List[List[float]], persist_directory: Path) -> SimpleVectorStore:
    """
    Create and persist a simple vector store.

    Args:
        documents: List of document dicts
        embeddings: List of embedding vectors
        persist_directory: Directory to persist the vector store

    Returns:
        SimpleVectorStore instance
    """
    store = SimpleVectorStore(persist_directory)
    store.add_documents(documents, embeddings)
    store.save()
    return store


def load_vector_store(persist_directory: Path) -> Optional[SimpleVectorStore]:
    """
    Load an existing vector store.

    Args:
        persist_directory: Directory where the vector store is persisted

    Returns:
        SimpleVectorStore instance or None if not found
    """
    store = SimpleVectorStore(persist_directory)
    if store.load():
        return store
    return None


def create_retriever(vector_store: SimpleVectorStore, k: int = RETRIEVAL_TOP_K):
    """
    Create a retriever function for the vector store.

    Args:
        vector_store: Vector store instance
        k: Number of results to return

    Returns:
        Retriever function
    """
    def retriever(query: str, embedding_model) -> List[Dict[str, Any]]:
        query_embedding = embedding_model.embed_query(query)
        return vector_store.similarity_search(query_embedding, k=k)
    return retriever
