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
            # 注意：PyPDFLoader 生成的 metadata 里有两个页码字段，不要混淆——
            #   page       : 0 基下标，供程序做索引（PDF 第 1 页 -> page=0）
            #   page_label : 1 基的展示标签，等于人翻 PDF 时看到的页码
            # 这里保持原样不做修改，展示场景由 rag_pipeline 负责挑选正确的字段。
            logger.info(f"Loaded {len(docs)} page(s) from {pdf_file.name}")
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {str(e)}")
            continue

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
