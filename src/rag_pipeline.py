from typing import Optional, List, Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) pipeline for question answering.
    """

    def __init__(
        self,
        llm,
        retriever,
        embedding_model,
        use_reranker: bool = False
    ):
        """
        Initialize the RAG pipeline.

        Args:
            llm: Language model instance
            retriever: Retriever function
            embedding_model: Embedding model instance
            use_reranker: Whether to use a reranker for improved retrieval
        """
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        logger.info("RAG pipeline initialized")

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            content = doc.get("page_content", "")

            # 展示给人和 LLM 的页码必须是 1 基的。PyPDFLoader 提供的 page_label
            # 正好是这个语义（PDF 第 1 页 -> "1"），优先使用；某些 loader 不带该
            # 字段，则回退到 0 基的 page + 1，避免出现 "Page: 0" 这种误读。
            page = metadata.get("page_label")
            if page is None:
                raw_page = metadata.get("page")
                page = raw_page + 1 if isinstance(raw_page, int) else "unknown"

            context_parts.append(f"[Document {i}]\nSource: {source}\nPage: {page}\nContent:\n{content}\n")
        return "\n".join(context_parts)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the RAG pipeline.

        Args:
            inputs: Dictionary containing 'question' key

        Returns:
            Dictionary with 'answer', 'context', and 'source_documents'
        """
        question = inputs.get("question", "")
        if not question:
            return {"answer": "Please provide a question.", "context": "", "source_documents": []}

        logger.info(f"Processing question: {question[:50]}...")

        retrieved_docs = self.retriever(question, self.embedding_model)

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            if self.llm:
                answer = self.llm.invoke(question).content
                return {"answer": answer, "context": "", "source_documents": []}
            else:
                return {"answer": "No relevant documents found. Please try a different question.", "context": "", "source_documents": []}

        context = self._format_context(retrieved_docs)

        if self.llm:
            prompt = f"""
            基于以下上下文信息回答问题。如果上下文没有相关信息，请回答"无法从提供的文档中找到答案"。

            上下文:
            {context}

            问题: {question}
            """

            logger.info("Generating answer using LLM")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        else:
            answer = "LLM not configured. Here are the relevant document snippets:\n\n" + context

        return {
            "answer": answer,
            "context": context,
            "source_documents": retrieved_docs
        }
