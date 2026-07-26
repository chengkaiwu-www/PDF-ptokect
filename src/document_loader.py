from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_pdf_documents(pdf_directory: Path) -> List[Document]:
    """
    Load all PDF documents from the specified directory.

    Args:
        pdf_directory: Path to the directory containing PDF files

    Returns:
        List of Document objects
    """
    if not pdf_directory.exists():
        logger.error(f"PDF directory does not exist: {pdf_directory}")
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")

    pdf_files = list(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) to load")

    documents = []
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} page(s) from {pdf_file.name}")
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {str(e)}")
            continue

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
