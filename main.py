import argparse
import sys
from pathlib import Path

from config.config import PDF_DIR, VECTOR_STORE_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_BASE
from utils.logger import setup_logger
from src.document_loader import load_pdf_documents
from src.text_splitter import split_documents
from src.embedding import get_embedding_model, create_embeddings
from src.vector_store import create_vector_store, load_vector_store, create_retriever
from src.rag_pipeline import RAGPipeline

logger = setup_logger(__name__)


def initialize_rag_system(recreate: bool = False):
    """
    Initialize the RAG system by loading documents and creating vector store.

    Args:
        recreate: Whether to recreate the vector store

    Returns:
        RAGPipeline instance
    """
    logger.info("Initializing RAG system")

    embedding_model = get_embedding_model()

    index_path = VECTOR_STORE_DIR / "index.json"
    if recreate or not index_path.exists():
        logger.info("Creating new vector store from PDF documents")
        documents = load_pdf_documents(PDF_DIR)

        if not documents:
            logger.error("No documents loaded. Please add PDF files to the data/pdfs directory.")
            sys.exit(1)

        split_docs = split_documents(documents)
        
        docs_dict = [{
            "page_content": doc.page_content,
            "metadata": doc.metadata
        } for doc in split_docs]

        texts = [doc["page_content"] for doc in docs_dict]
        embeddings = create_embeddings(texts, embedding_model)

        vector_store = create_vector_store(docs_dict, embeddings, VECTOR_STORE_DIR)
    else:
        logger.info("Loading existing vector store")
        vector_store = load_vector_store(VECTOR_STORE_DIR)

        if vector_store is None:
            logger.warning("Failed to load vector store, recreating from scratch")
            documents = load_pdf_documents(PDF_DIR)
            if not documents:
                logger.error("No documents available")
                sys.exit(1)
            split_docs = split_documents(documents)
            
            docs_dict = [{
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in split_docs]

            texts = [doc["page_content"] for doc in docs_dict]
            embeddings = create_embeddings(texts, embedding_model)

            vector_store = create_vector_store(docs_dict, embeddings, VECTOR_STORE_DIR)

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
    parser.add_argument("--recreate", action="store_true", help="Recreate the vector store")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--query", type=str, help="Query to answer")

    args = parser.parse_args()

    if not args.chat and not args.query:
        parser.print_help()
        sys.exit(1)

    try:
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
