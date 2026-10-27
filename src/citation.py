"""
引用解析与合法性校验（纯确定性，不调用任何模型）。

## 为什么引用要能自动校验

RAG 的「可信」不是靠 prompt 说"请标注来源"，而是靠**引用可以被机器验证**：
答案里说 `[来源: pumpkin-book.pdf 第31页]`，那这次召回里必须真的存在这一页的块。
否则就是**编造引用**——比没有引用更危险，因为它给了读者虚假的可核查感。

因此把「格式约定」和「校验逻辑」放在同一处：
    prompts/qa_prompt.py 规定格式  ->  本模块解析并校验
两者一旦不同步，指标立刻失真，所以不能分散在两边各自实现。

## 三种错误要分开统计

| 指标 | 含义 | 说明 |
|---|---|---|
| 引用覆盖率 | 有引用的答案占比 | 衡量「愿不愿意标」 |
| 引用合法率 | 合法引用 / 全部引用 | 衡量「标得对不对」 |
| 无引用答案 | 完全没标引用的答案数 | 这类答案无法溯源，即使内容正确也不合格 |

把三者合成一个分数会掩盖问题：例如覆盖率 100% 但合法率 40%，
说明模型在**自信地编造**；反过来覆盖率 40% 合法率 100%，
说明模型偏保守。两种情况的修法完全不同。
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

# 兼容半角/全角括号与冒号；页码允许前后有空格
CITATION_PATTERN = re.compile(
    r"[\[【]\s*来源\s*[:：]\s*(?P<file>[^\[\]【】\s]+?)\s*第\s*(?P<page>\d+)\s*页\s*[\]】]"
)


def _norm_page(value: Any) -> str:
    """页码归一化：'03' / 3 / '3' 都视为 3。"""
    text = str(value).strip()
    return str(int(text)) if text.isdigit() else text


def _doc_key(doc: Dict[str, Any]) -> tuple:
    meta = doc.get("metadata", {})
    source = Path(meta.get("source", "")).name
    page = meta.get("page_label")
    if page is None:
        raw = meta.get("page")
        page = raw + 1 if isinstance(raw, int) else "unknown"
    return source, _norm_page(page)


def parse_citations(answer: str) -> List[Dict[str, str]]:
    """抽出答案里的所有引用标记（按出现顺序，允许重复）。"""
    return [
        {"file": m.group("file").strip(), "page": _norm_page(m.group("page"))}
        for m in CITATION_PATTERN.finditer(answer or "")
    ]


def validate_citations(answer: str, documents: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    校验答案中的引用是否都能对应到本次召回的块。

    Args:
        answer: 模型输出的答案
        documents: 本次召回的块（含 metadata.source / page_label）

    Returns:
        {
          "count": 引用总数,
          "valid": 合法引用数,
          "invalid": [非法引用...],
          "validity": 合法率（无引用时为 None，避免"没引用=全对"的假象）,
          "has_citation": 是否至少有一个引用,
          "cited_sources": 被引用到的文件名去重列表,
        }
    """
    allowed = {_doc_key(d) for d in documents}
    citations = parse_citations(answer)

    invalid = [c for c in citations if (c["file"], c["page"]) not in allowed]
    cited_sources = sorted({c["file"] for c in citations if (c["file"], c["page"]) in allowed})

    return {
        "count": len(citations),
        "valid": len(citations) - len(invalid),
        "invalid": invalid,
        "validity": (len(citations) - len(invalid)) / len(citations) if citations else None,
        "has_citation": bool(citations),
        "cited_sources": cited_sources,
    }
