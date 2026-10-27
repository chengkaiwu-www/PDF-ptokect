import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from prompts.qa_prompt import (
    PROMPT_VARIANTS,
    REFUSAL_SENTINEL,
    REFUSAL_TEXT,
    build_prompt_variant,
)
from src.citation import validate_citations
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGPipeline:
    """
    RAG 编排：检索 -> 组装 context -> 调 LLM 生成 -> 校验引用。

    **职责边界（S4 明确）**：本类不关心「怎么检索」——那是 retriever 的事。
    S0~S3 这里曾挂着一个 `use_reranker` 参数，声明了却从未生效，
    因为它把「检索策略」和「生成编排」混在了一层。
    S4 把精排移到检索层（`HybridRetriever(reranker=...)`），本类只负责编排。

    **S5 新增的职责**：生成不是「把 context 丢给模型」就完事，
    还要保证输出**可溯源**。所以本类在生成后做两件事：
      1. 识别拒答哨兵，换成给人看的自然语言；
      2. 解析引用标记并与本次召回块比对，把「引用合法率」作为返回值带出去。
    第 2 步刻意不用 LLM —— 它是确定性问题，用模型判断反而引入不确定性。
    """

    def __init__(
        self,
        llm,
        retriever,
        embedding_model,
        top_k: Optional[int] = None,
        prompt_variant: str = "full",
    ):
        """
        Args:
            llm: 语言模型实例（可为 None，此时只跑检索）
            retriever: 可调用对象，签名 (query, embedding_model) -> List[dict]
            embedding_model: embedding 模型实例
            top_k: 传给 retriever 的召回条数；None 表示用 retriever 自身默认值
            prompt_variant: full | no_refusal | no_cite，仅供消融实验使用
        """
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.prompt_variant = prompt_variant
        # 版本号跟着变体走：评测结果里带版本号，不同 prompt 的数字才不会混表
        self.prompt_version = PROMPT_VARIANTS[prompt_variant][1]

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        把召回的块拼成 context。

        给模型的 Source 必须是**文件名**而不是完整路径：
          1. 路径里的反斜杠、盘符与中文目录会干扰模型复述引用格式；
          2. 校验时也要按文件名比对，两边用同一个键才不会出现「模型照抄了
             完整路径，校验却只认文件名」的假不匹配。
        Page 用 1 基的 page_label（loader 提供），缺失时回退 page + 1。
        """
        parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            source = Path(metadata.get("source", "unknown")).name
            content = doc.get("page_content", "")

            page = metadata.get("page_label")
            if page is None:
                raw_page = metadata.get("page")
                page = raw_page + 1 if isinstance(raw_page, int) else "unknown"

            parts.append(
                f"[Document {i}]\nSource: {source}\nPage: {page}\nContent:\n{content}\n"
            )
        return "\n".join(parts)

    def _retrieve(self, question: str) -> List[Dict[str, Any]]:
        if self.top_k is None:
            return self.retriever(question, self.embedding_model)
        try:
            return self.retriever(question, self.embedding_model, top_k=self.top_k)
        except TypeError:
            # 兼容只有两个参数的旧闭包（vector_store.create_retriever）
            return self.retriever(question, self.embedding_model)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """
        判定是否拒答。

        只看开头 40 个字符：模型偶尔会在哨兵前后加一句客套话，
        这时仍应算作拒答；但如果哨兵出现在答案末尾，说明它**先答了再拒**，
        那是矛盾输出，不能算合格拒答。
        """
        return REFUSAL_SENTINEL in (text or "")[:40].upper()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        跑完整流程。

        Returns:
            {
              "answer": 展示用答案（拒答已被替换成自然语言）,
              "raw_answer": 模型原始输出（排查用）,
              "refused": 是否判定为拒答,
              "context": 拼好的上下文,
              "source_documents": 召回块,
              "citations": 引用校验结果（见 src/citation.py）,
              "prompt_version": prompt 版本号,
            }
        """
        question = inputs.get("question", "")
        empty = {
            "answer": "Please provide a question.",
            "raw_answer": "",
            "refused": True,
            "context": "",
            "source_documents": [],
            "citations": validate_citations("", []),
            "prompt_version": self.prompt_version,
        }
        if not question:
            return empty

        logger.info(f"Processing question: {question[:50]}...")

        retrieved_docs = self._retrieve(question)

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            # 没有召回就**不该**调 LLM 硬答：那等于让模型凭参数记忆回答，
            # 而这正是 RAG 要避免的幻觉来源。直接拒答，且不花 token。
            return {
                "answer": REFUSAL_TEXT,
                "raw_answer": REFUSAL_SENTINEL,
                "refused": True,
                "context": "",
                "source_documents": [],
                "citations": validate_citations("", []),
                "prompt_version": self.prompt_version,
            }

        context = self._format_context(retrieved_docs)

        if not self.llm:
            # 没有 LLM 时退化为「只展示召回片段」，此时不存在生成层指标
            return {
                "answer": "LLM not configured. Here are the relevant document snippets:\n\n" + context,
                "raw_answer": "",
                "refused": False,
                "context": context,
                "source_documents": retrieved_docs,
                "citations": {"count": 0, "valid": 0, "invalid": [], "validity": None,
                              "has_citation": False, "cited_sources": []},
                "prompt_version": self.prompt_version,
            }

        prompt, version = build_prompt_variant(question, context, self.prompt_variant)
        logger.info(f"Generating answer using LLM (prompt={version})")
        response = self.llm.invoke(prompt)
        raw_answer = response.content if hasattr(response, "content") else str(response)
        raw_answer = (raw_answer or "").strip()

        refused = self._is_refusal(raw_answer)
        citations = validate_citations("" if refused else raw_answer, retrieved_docs)

        return {
            "answer": REFUSAL_TEXT if refused else raw_answer,
            "raw_answer": raw_answer,
            "refused": refused,
            "context": context,
            "source_documents": retrieved_docs,
            "citations": citations,
            "prompt_version": self.prompt_version,
        }

    # ---------------------------------------------------------------- 流式
    def stream(
        self,
        question: str,
        documents: Optional[List[Dict[str, Any]]] = None,
        retrieve_ms: Optional[float] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式版本：**先给来源，再逐 token 给答案，最后给引用校验结果**。

        为什么另开一个方法，而不是让 `invoke()` 支持 stream：
          `invoke()` 的语义是「一次拿到完整结果」，评测脚本依赖这个确定性；
          流式只在交互场景有价值。把两种语义塞进一个函数，评测侧就得多一层分支
          去猜「这次拿到的是完整答案还是半截」。**同一份逻辑，两种接口，各自单纯。**

        两个细节：

        (1) **来源先于答案**。检索完立刻把块发出去，用户能在等待生成时先看到
            「参考资料是哪几页」——这也是把「检索」和「生成」两层的耗时分开呈现，
            排查「答得慢」时能一眼看出是检索慢还是模型慢。

        (2) **哨兵遮蔽**。拒答时模型输出 `INSUFFICIENT_CONTEXT`，这是给机器看的信号，
            不该出现在聊天界面上。所以要 hold 住开头若干字符再决定：
              · 与哨兵前缀一致      -> 继续等（还判断不出）
              · 确认是哨兵          -> 一个字都不发，只在 done 里标记 refused
              · 一旦不匹配          -> 立刻把攒着的字符一起发出去（延迟约几十毫秒）
            这比「先发出去再撤回」可靠，代价只是开头几十毫秒不显示。

        Args:
            documents: 可选的**预检索结果**。服务端为了拿到每块的通道名次（调试信息）
                已经检索过一次，传进来避免为展示来源再检索第二遍 —— 同一份数据
                不该算两次、更不该出现两次结果不一致。
            retrieve_ms: 配套的检索耗时；未传则按 0 计。
        """
        question = (question or "").strip()
        empty_citations = validate_citations("", [])

        if not question:
            yield {
                "type": "sources", "documents": [], "retrieve_ms": 0.0,
            }
            yield {
                "type": "done", "answer": "Please provide a question.", "raw_answer": "",
                "refused": True, "citations": empty_citations, "gen_ms": 0.0,
                "prompt_version": self.prompt_version,
            }
            return

        t0 = time.time()
        if documents is None:
            retrieved_docs = self._retrieve(question)
            retrieve_ms = round((time.time() - t0) * 1000, 1)
        else:
            retrieved_docs = documents
            retrieve_ms = float(retrieve_ms or 0.0)

        yield {"type": "sources", "documents": retrieved_docs, "retrieve_ms": retrieve_ms}

        if not retrieved_docs:
            logger.warning("No documents retrieved")
            yield {
                "type": "done", "answer": REFUSAL_TEXT, "raw_answer": REFUSAL_SENTINEL,
                "refused": True, "citations": empty_citations, "gen_ms": 0.0,
                "prompt_version": self.prompt_version,
            }
            return

        context = self._format_context(retrieved_docs)

        if not self.llm:
            yield {
                "type": "done",
                "answer": "LLM not configured. Here are the relevant document snippets:\n\n" + context,
                "raw_answer": "", "refused": False,
                "citations": {"count": 0, "valid": 0, "invalid": [], "validity": None,
                              "has_citation": False, "cited_sources": []},
                "gen_ms": 0.0, "prompt_version": self.prompt_version,
            }
            return

        prompt, version = build_prompt_variant(question, context, self.prompt_variant)
        g0 = time.time()

        raw = ""          # 模型原始输出（用于引用校验，不受遮蔽影响）
        pending = ""      # 已收到但尚未决定要不要发的字符
        started = False   # 是否已开始向界面发送
        refused = False

        for chunk in self.llm.stream(prompt):
            text = getattr(chunk, "content", "") or ""
            if not text:
                continue
            raw += text

            if refused:
                continue

            if not started:
                pending += text
                head = pending.lstrip()
                looks_like_sentinel = head.upper()[: len(REFUSAL_SENTINEL)] == REFUSAL_SENTINEL[: len(head)]
                if looks_like_sentinel and len(head) <= len(REFUSAL_SENTINEL):
                    continue                      # 还可能是哨兵，再等等
                if head.upper().startswith(REFUSAL_SENTINEL):
                    refused = True                # 确认拒答，一个字都不发
                    pending = ""
                    continue
                started = True
                yield {"type": "token", "text": pending}
                pending = ""
            else:
                yield {"type": "token", "text": text}

        gen_ms = round((time.time() - g0) * 1000, 1)

        if not refused and not started and pending:
            # 输出极短（如一句「资料中未提及」），流结束时才落地
            if pending.lstrip().upper().startswith(REFUSAL_SENTINEL):
                refused = True
            else:
                started = True
                yield {"type": "token", "text": pending}

        raw = raw.strip()
        if not refused:
            # 兜底：与 invoke() 用同一个判定函数，避免两条路径口径漂移
            refused = self._is_refusal(raw)

        citations = validate_citations("" if refused else raw, retrieved_docs)

        yield {
            "type": "done",
            "answer": REFUSAL_TEXT if refused else raw,
            "raw_answer": raw,
            "refused": refused,
            "citations": citations,
            "gen_ms": gen_ms,
            "retrieve_ms": retrieve_ms,
            "prompt_version": version,
        }
