from typing import List
from pathlib import Path

import pypdfium2 as pdfium
from langchain_core.documents import Document

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _load_single_pdf(pdf_path: Path) -> List[Document]:
    """
    用 pypdfium2（PDFium 引擎）逐页抽取文字。

    为什么不用 langchain 的 PyPDFLoader？——它的底层是 pypdf，对中文 PDF 存在
    抽取失败的实测案例（见 README「实验 3：PDF 解析器选型」）。同一份中文 PDF：
        pypdf      -> 汉字占比 0.0%（乱码）
        pypdfium2  -> 汉字占比 27.6%（可用），且速度快 8~16 倍
    根因是 PDF 内嵌字体子集缺少 ToUnicode 映射，不同解析器的兜底策略不同。

    metadata 约定：
        source      : PDF 绝对路径
        page        : 0 基下标，供程序做索引（PDF 第 1 页 -> page=0）
        page_label  : 1 基展示页码，等于人翻 PDF 时看到的页码
        total_pages : 该 PDF 总页数
    """
    docs: List[Document] = []
    empty_pages = 0

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(pdf)
        for page_index in range(total_pages):
            textpage = pdf[page_index].get_textpage()
            text = textpage.get_text_range() or ""
            # PDFium 用 \r\n 分行，统一成 \n，避免下游正则和切分逻辑要多套一份
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if not text.strip():
                empty_pages += 1
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_index,
                    "page_label": str(page_index + 1),
                    "total_pages": total_pages,
                },
            ))
    finally:
        pdf.close()

    if empty_pages:
        # 扫描版 PDF 抽不出文字，是 RAG 里最常见的「静默失败」：不报错，但检索永远为空
        logger.warning(
            f"{pdf_path.name}: {empty_pages}/{len(docs)} page(s) have no extractable text "
            f"(扫描版 PDF 需要 OCR，否则这些页面对检索完全不可见)"
        )

    return docs


def load_pdf_documents(pdf_directory: Path) -> List[Document]:
    """
    Load all PDF documents from the specified directory.

    Args:
        pdf_directory: Path to the directory containing PDF files

    Returns:
        List of Document objects (one per page)
    """
    if not pdf_directory.exists():
        logger.error(f"PDF directory does not exist: {pdf_directory}")
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")

    pdf_files = sorted(pdf_directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) to load")

    documents: List[Document] = []
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")
            docs = _load_single_pdf(pdf_file)
            chars = sum(len(d.page_content) for d in docs)
            logger.info(f"Loaded {len(docs)} page(s) / {chars} chars from {pdf_file.name}")
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {str(e)}")
            continue

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
