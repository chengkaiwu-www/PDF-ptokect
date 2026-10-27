"""
生成层 prompt 模板（版本化）。

## 为什么要把 prompt 抽成模块

S0~S4 的 prompt 是写在 `rag_pipeline.invoke()` 里的一个 f-string，带来两个问题：

1. **无法版本化**。评测要看「prompt 改动带来什么影响」，但 f-string 改了就没了，
   历史结果无法归因。这里用 PROMPT_VERSION 显式标版本，评测结果里一并记录。
2. **约束写不进去**。原来只有一句「基于上下文回答，不知道就说不知道」，
   模型基本会忽略。真正有效的 prompt 需要把「不能用什么知识」「引用什么格式」
   「不满足条件时输出什么」写成**可机械校验**的条款。

## 三个关键设计

**(1) 强制引用格式** —— `[来源: 文件名 第N页]`
    格式必须**唯一且可正则解析**，否则无法自动计算「引用合法率」。
    要求模型引用「文件名 + 页码」而不是「第几段」，是因为前者能被独立验证：
    页码必须真的出现在本次召回的块里。这让「编造引用」变成可检测的错误。

**(2) 拒答哨兵 `INSUFFICIENT_CONTEXT`**
    不用自然语言句子的原因：模型会改写（"抱歉，资料中未提及…"），
    正则匹配永远漏。固定哨兵是机器可判定的；展示给人时再换成自然语言
    （见 RAGPipeline.REFUSAL_TEXT）。**机器可读与人类可读分开**。

**(3) 明确禁止资料外知识**
    这是 RAG 与「带检索的聊天」的分界线。实测中它对
    「语料没有但模型知道」的题（如 BM25、HNSW）效果有限 —— 见 README 实验 8。
"""

PROMPT_VERSION = "v1-cite"

# 机器可判定的拒答哨兵
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

# 展示给人看的拒答话术（不进 prompt）
REFUSAL_TEXT = "资料中没有找到足够的信息来回答这个问题。"

QA_PROMPT_TEMPLATE = """你是文档问答助手。只能依据下面的【资料】回答，不得使用资料之外的知识。

【资料】
{context}

【问题】
{question}

【回答要求】
1. 只使用【资料】中的内容作答。资料没提到的，不要推测、不要补充常识。
2. 每一个事实性结论后面必须紧跟引用标记，格式严格为：[来源: 文件名 第N页]
   文件名与页码必须来自【资料】里给出的 Source 与 Page，不要自己编造。
3. 一个结论可以由多处资料支撑时，连续写多个标记，例如：
   [来源: a.pdf 第3页][来源: b.pdf 第12页]
4. 如果【资料】不足以回答问题，只输出这一行，不要输出任何其它内容：
   INSUFFICIENT_CONTEXT
5. 用中文回答，简洁直接，不要复述问题，不要写"根据资料"这类套话。

【回答】
"""


def build_prompt(question: str, context: str) -> str:
    return QA_PROMPT_TEMPLATE.format(question=question, context=context)


# ---------------------------------------------------------------------------
# 消融用的 prompt 变体
#
# 目的不是「提供选项」，而是回答两个具体问题：
#   1. 拒答是模型自觉，还是 prompt 条款起作用？  -> no_refusal
#   2. 引用是模型自觉，还是格式要求起作用？      -> no_cite
# 没有这两个变体，「我们加了引用和拒答约束」就只是一句自我声明，无法证明。
#
# 注意：变体会改变 PROMPT_VERSION，评测结果必须带上版本号，
# 否则不同 prompt 的数字会被混在一张表里比较。
# ---------------------------------------------------------------------------

_NO_REFUSAL_RULE = """4. 如果【资料】信息不完整，也要基于资料中相关的内容给出最接近的回答，不要拒答。"""

_REFUSAL_RULE = """4. 如果【资料】不足以回答问题，只输出这一行，不要输出任何其它内容：
   INSUFFICIENT_CONTEXT"""

_CITE_RULES = """2. 每一个事实性结论后面必须紧跟引用标记，格式严格为：[来源: 文件名 第N页]
   文件名与页码必须来自【资料】里给出的 Source 与 Page，不要自己编造。
3. 一个结论可以由多处资料支撑时，连续写多个标记，例如：
   [来源: a.pdf 第3页][来源: b.pdf 第12页]"""

_NO_CITE_RULES = """2. 正常作答即可，不需要标注来源。"""

QA_PROMPT_NO_REFUSAL = QA_PROMPT_TEMPLATE.replace(_REFUSAL_RULE, _NO_REFUSAL_RULE)

QA_PROMPT_NO_CITE = QA_PROMPT_TEMPLATE.replace(_CITE_RULES, _NO_CITE_RULES)

PROMPT_VARIANTS = {
    "full": (QA_PROMPT_TEMPLATE, PROMPT_VERSION),
    "no_refusal": (QA_PROMPT_NO_REFUSAL, "v1-norefuse"),
    "no_cite": (QA_PROMPT_NO_CITE, "v1-nocite"),
}


def build_prompt_variant(question: str, context: str, variant: str = "full"):
    """返回 (prompt, version)。version 供评测记录口径。"""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"未知 prompt 变体: {variant}（可选 {list(PROMPT_VARIANTS)}）")
    template, version = PROMPT_VARIANTS[variant]
    return template.format(question=question, context=context), version
