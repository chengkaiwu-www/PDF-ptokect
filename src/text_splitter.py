from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import setup_logger

logger = setup_logger(__name__)


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    Args:
        documents: List of Document objects to split
        CHUNK_SIZE: Maximum size of each chunk
        CHUNK_OVERLAP: Overlap between consecutive chunks

    Returns:
        List of split Document objects
    """
    if not documents:
        logger.warning("No documents to split")
        return []

    logger.info(f"Splitting {len(documents)} document(s) with chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    split_docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunk(s)")

    return split_docs
