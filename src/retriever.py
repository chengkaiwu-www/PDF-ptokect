from typing import Optional
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from config.config import RETRIEVAL_TOP_K
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_retriever(
    vector_store: Chroma,
    search_type: str = "similarity",
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: Optional[float] = None
) -> VectorStoreRetriever:
    """
    Create a retriever from a vector store.

    Args:
        vector_store: Chroma vector store instance
        search_type: Type of search (similarity, similarity_score_threshold, mmr)
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score threshold (for similarity_score_threshold search)

    Returns:
        VectorStoreRetriever instance
    """
    if vector_store is None:
        logger.error("Vector store is None, cannot create retriever")
        raise ValueError("Vector store cannot be None")

    logger.info(f"Creating retriever with search_type={search_type}, top_k={top_k}")

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": top_k,
            "score_threshold": score_threshold
        } if score_threshold else {"k": top_k}
    )

    logger.info("Retriever created successfully")
    return retriever


class CustomRetriever(VectorStoreRetriever):
    """
    Custom retriever with additional filtering capabilities.
    """

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        logger.info(f"Custom retrieval for query: {query}")
        return super()._get_relevant_documents(query, run_manager=run_manager)
