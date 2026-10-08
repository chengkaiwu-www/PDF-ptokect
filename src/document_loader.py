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

            # PyPDFLoader 写入的 metadata["page"] 是 0 基下标（PDF 第 1 页 -> page=0），
            # 直接展示会显示 "Page: 0"。在「PDF -> Document」这个边界上统一规范化为
            # 1 基页码，下游（展示、过滤、引用溯源）拿到的就都是人类可读的页码，
            # 避免每一处消费方各写一次 +1（off-by-one 是最容易埋雷的一类 bug）。
            for doc in docs:
                if isinstance(doc.metadata.get("page"), int):
                    doc.metadata["page"] += 1

            logger.info(f"Loaded {len(docs)} page(s) from {pdf_file.name}")
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {str(e)}")
            continue

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
