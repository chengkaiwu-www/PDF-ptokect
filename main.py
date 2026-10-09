import argparse
import sys
from typing import Any, Dict, List

from config.config import (
    PDF_DIR,
    VECTOR_STORE_DIR,
    VECTOR_BACKEND,
    RETRIEVAL_TOP_K,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_API_BASE,
)
from utils.logger import setup_logger
from src.document_loader import load_pdf_file
from src.text_splitter import split_documents
from src.embedding import get_embedding_model, create_embeddings
from src.vector_store import (
    DOCUMENTS_FILE,
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    create_vector_store,
    load_vector_store,
    create_retriever,
)
from src.vector_utils import file_fingerprint
from src.rag_pipeline import RAGPipeline

logger = setup_logger(__name__)


def _clear_index_artifacts() -> None:
    """--recreate 时清掉旧的派生文件，避免换了后端还留着上一个后端的残留索引。"""
    for name in (DOCUMENTS_FILE, EMBEDDINGS_FILE, FAISS_INDEX_FILE):
        path = VECTOR_STORE_DIR / name
        if path.exists():
            path.unlink()
            logger.info(f"已删除旧的索引文件 {name}")


def sync_vector_store(embedding_model, recreate: bool = False):
    """
    让向量库与 data/pdfs 保持一致：**只处理新增或内容变动的 PDF**。

    这是 S3 的「增量索引」：
      · 每个 PDF 记一份内容指纹（sha1），指纹没变就跳过 —— 不重复解析、不重复 embedding；
      · chunk 层面还有内容 sha1 去重（在 VectorStore.add_documents 里），
        它能兜住「两个 PDF 含相同章节」「同一文件被重复入库」这类情况。

    与「每次全量重建」相比，代价是只增不删：删掉某个 PDF 后，它的 chunk 仍在库里
    （FAISS 不支持按文档删除），需要 --recreate 才能清干净。这一点在日志里会明确提示。
    """
    if recreate:
        _clear_index_artifacts()

    store = None if recreate else load_vector_store(VECTOR_STORE_DIR)

    if store is None:
        logger.info("未找到可用索引，从头构建")
        store = create_vector_store([], [], VECTOR_STORE_DIR)
        indexed: Dict[str, Any] = {}
    else:
        indexed = dict((store.meta or {}).get("indexed_files") or {})

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"{PDF_DIR} 下没有 PDF 文件")
        return store

    pending: List = []
    for pdf in pdf_files:
        fingerprint = file_fingerprint(pdf)
        if indexed.get(pdf.name, {}).get("sha1") == fingerprint:
            continue
        pending.append((pdf, fingerprint))

    if not pending:
        if store.needs_persist():
            # 内容没变，但后端/索引类型变了（内存里已重建，磁盘上还没有）——补一次落盘
            logger.info(f"内容无变化，但后端/索引类型有变，重新落盘（{VECTOR_BACKEND}）")
            store.save()
        else:
            logger.info(f"索引已是最新：{len(pdf_files)} 个 PDF，{len(store)} 个 chunk")
        return store

    logger.info(f"发现 {len(pending)} 个待索引 PDF（共 {len(pdf_files)} 个）")
    for pdf, fingerprint in pending:
        docs = load_pdf_file(pdf)
        if not docs:
            logger.warning(f"{pdf.name} 没有解析出任何内容，跳过")
            continue

        chunks = split_documents(docs)
        payload = [
            {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
        ]
        embeddings = create_embeddings([p["page_content"] for p in payload], embedding_model)
        added = store.add_documents(payload, embeddings)
        indexed[pdf.name] = {"sha1": fingerprint, "chunks": len(chunks), "added": added}
        logger.info(f"{pdf.name}: {len(chunks)} chunk，实际新增 {added}")

    missing = [name for name in indexed if name not in {p.name for p in pdf_files}]
    if missing:
        logger.warning(
            f"{len(missing)} 个曾索引过的 PDF 已从 {PDF_DIR.name} 中移除，"
            f"但它们的 chunk 仍留在向量库里（需要 --recreate 才能彻底清除）"
        )

    store.meta["indexed_files"] = indexed
    store.save()
    return store


def build_vector_store(embedding_model, recreate: bool = False):
    """同步（或重建）向量库，并保证结果非空。"""
    vector_store = sync_vector_store(embedding_model, recreate=recreate)
    if len(vector_store) == 0:
        logger.error(f"向量库为空。请把 PDF 放进 {PDF_DIR}，再执行 main.py --recreate")
        sys.exit(1)
    return vector_store


def initialize_rag_system(recreate: bool = False):
    """
    Initialize the RAG system by loading documents and creating vector store.

    Args:
        recreate: Whether to rebuild the vector store from scratch

    Returns:
        RAGPipeline instance
    """
    logger.info(f"Initializing RAG system (backend={VECTOR_BACKEND}, top_k={RETRIEVAL_TOP_K})")

    embedding_model = get_embedding_model()
    vector_store = build_vector_store(embedding_model, recreate=recreate)

    retriever = create_retriever(vector_store)

    if DEEPSEEK_API_KEY:
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE,
            temperature=0
        )
        logger.info(f"DeepSeek LLM initialized: {DEEPSEEK_MODEL}")
    else:
        logger.warning("DEEPSEEK_API_KEY not found. Please set it in the .env file to enable LLM generation.")
        logger.info("You can still use the retrieval functionality.")
        llm = None

    rag_pipeline = RAGPipeline(llm=llm, retriever=retriever, embedding_model=embedding_model)

    return rag_pipeline


def chat_mode(rag_pipeline):
    """
    Interactive chat mode for the RAG system.

    Args:
        rag_pipeline: RAGPipeline instance
    """
    logger.info("Starting chat mode. Type 'exit' to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            result = rag_pipeline.invoke({"question": question})

            print(f"\nAnswer: {result['answer']}")
            print(f"\nSource Documents: {len(result['source_documents'])}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="PDF RAG System with DeepSeek")
    parser.add_argument("--recreate", action="store_true",
                        help="丢弃现有索引并全量重建（默认走增量：只索引新增/变动的 PDF）")
    parser.add_argument("--reindex", action="store_true",
                        help="只同步/重建索引后退出，不进入问答")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--query", type=str, help="Query to answer")

    args = parser.parse_args()

    if not args.chat and not args.query and not args.reindex:
        parser.print_help()
        sys.exit(1)

    try:
        if args.reindex:
            embedding_model = get_embedding_model()
            store = build_vector_store(embedding_model, recreate=args.recreate)
            print(
                f"\n索引已就绪：{len(store)} chunk，后端 {store.backend_name}"
                f"/{store.meta.get('index_type', '-')}，维度 {store.meta.get('dim')}"
            )
            return

        rag_pipeline = initialize_rag_system(recreate=args.recreate)

        if args.query:
            result = rag_pipeline.invoke({"question": args.query})
            print(f"\nAnswer: {result['answer']}")
            print(f"\nContext:\n{result['context']}")

        if args.chat:
            chat_mode(rag_pipeline)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
