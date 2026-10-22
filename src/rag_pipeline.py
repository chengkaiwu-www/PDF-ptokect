from typing import Any, Dict, List, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGPipeline:
    """
    RAG 编排：检索 -> 组装 context -> 调 LLM 生成。

    **职责边界（S4 明确）**：本类不关心「怎么检索」——那是 retriever 的事。
    S0~S3 这里曾挂着一个 `use_reranker` 参数，声明了却从未生效，
    因为它把「检索策略」和「生成编排」混在了一层。
    S4 把精排移到检索层（`HybridRetriever(reranker=...)`），本类只负责编排。
    这样做的实际好处：做检索 A/B 时不用碰生成代码，反之亦然。
    """

    def __init__(
        self,
        llm,
        retriever,
        embedding_model,
        top_k: Optional[int] = None,
    ):
        """
        Args:
            llm: 语言模型实例（可为 None，此时只跑检索）
            retriever: 可调用对象，签名 (query, embedding_model) -> List[dict]
            embedding_model: embedding 模型实例
            top_k: 传给 retriever 的召回条数；None 表示用 retriever 自身默认值
        """
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.top_k = top_k

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

            # 展示给人和 LLM 的页码必须是 1 基的。loader 提供的 page_label
            # 正好是这个语义（PDF 第 1 页 -> "1"），优先使用；若某个 loader 不带该
            # 字段，则回退到 0 基的 page + 1，避免出现 "Page: 0" 这种误读。
            page = metadata.get("page_label")
            if page is None:
                raw_page = metadata.get("page")
                page = raw_page + 1 if isinstance(raw_page, int) else "unknown"

            context_parts.append(f"[Document {i}]\nSource: {source}\nPage: {page}\nContent:\n{content}\n")
        return "\n".join(context_parts)

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        if self.top_k is None:
            return self.retriever(question, self.embedding_model)
        try:
            return self.retriever(question, self.embedding_model, top_k=self.top_k)
        except TypeError:
            # 兼容只有两个参数的旧闭包（vector_store.create_retriever）
            return self.retriever(question, self.embedding_model)

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

        retrieved_docs = self._retrieve(question)

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
