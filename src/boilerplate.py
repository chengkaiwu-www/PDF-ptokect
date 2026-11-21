"""识别并移除 PDF 里的「高分噪声」：目录页与页眉页脚。

为什么单独成模块
----------------
这类检测有两个使用者：
  · `scripts/detect_boilerplate.py` —— 只出清单，供人工抽查
  · `src/text_splitter.py`        —— 真正执行过滤
如果两处各写一份正则，迟早漂移：脚本说「4 页是目录」，切分器却按另一套规则删掉 100 页。
所以规则只在这里定义一次。

两类噪声的机制不同，不能用同一套判据
------------------------------------
· **目录页**：把全书章节标题聚在一起 → 关键词全中、但不含答案。
  向量检索会因为它「主题高度覆盖」而给高分，LLM 判定也会认为「这块能回答」。
  特征是「标题 + 点引导符 + 页码」，判据要卡在**点引导符后面跟页码**上。

· **页眉页脚**：每页重复印的书名、推广语、网址。
  特征是「同一本书内，一行出现在大比例页里」。
  它不制造假阳性，但会给每个块注入同一批高频词，稀释块之间的区分度。

第一版判据的教训
----------------
最初用「行尾页码 ≥6 且短行占比 ≥0.6」判目录页，结果南瓜书 129/196 页被判成目录
（占全库字符 35.9%）—— 它其实是公式详解，公式行天然以数字结尾、天然很短。
**「看起来像噪声」和「真的是噪声」是两件事，判据宽一格就会误伤整本书。**
现在只用「点引导符+页码」这一条强信号，宁可漏判不可误伤。

过滤之后：为什么默认关闭
------------------------
实测（README 实验 13）两个开关**都没有收益**，所以默认全关：
  · 目录页全库只有 4 页 / 53 块，在 140 个召回位置里只占 1 个（0.7%）。
    删掉后 Hit@1 四组不变、Precision 变化 ±1 块且方向不一致 —— 影响面太小，测不出来。
  · 删页眉的收益是去掉 2 行重复文本，**代价是重排整本书的切分边界**：
    每页少约 60 字符，384 token 的切分点整体前移，南瓜书 732 → 689 块，
    全库 32.5% 的块文本与基线不同 → 相关块净 -3，三组 Precision 一致下降。

这暴露了切分器的一个性质：**它会把「极小的文本改动」放大成「整本书的结构重组」。**
所以噪声更适合在**判定/报告层**标注（目录页只说明 Precision 是下界），
而不是从索引里删掉。代码保留，用于复现这个负结果。
"""
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from langchain_core.documents import Document

# 「标题 .......... 12」：点引导符后必须跟页码。
# 单看「有省略号」会误伤正文 —— 中文省略号「……」正好是两个 …，正文里极常见。
TOC_ENTRY = re.compile(
    r"(?:\.{3,}|…{2,}|·{3,}|-{4,}|_{4,})\s*(?:\d{1,4}|[ivxlcdm]{1,6})\s*$", re.I
)

TOC_MIN_ENTRIES = 3        # 至少 3 条「点引导符+页码」才认为是目录页
TOC_MIN_LINES = 8          # 太短的页不判（防止把正文中的小列表误判）
REPEAT_RATIO = 0.30        # 一行出现在 30% 以上的页 → 页眉页脚候选
REPEAT_MIN_PAGES = 3       # 至少 3 页，避免小文档误判
REPEAT_MIN_LEN = 4         # 太短的行（单个字符、页码）不参与统计，删了也没意义
REPEAT_MAX_LEN = 60        # 太长的行更可能是正文段落里的重复句，不删


def toc_entry_count(text: str) -> int:
    """数一页里有几条「点引导符 + 页码」结尾的行。"""
    return sum(1 for ln in text.splitlines() if TOC_ENTRY.search(ln.strip()))


def is_toc_page(text: str) -> bool:
    """整页判定：条目够多、页也够长。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < TOC_MIN_LINES:
        return False
    return toc_entry_count(text) >= TOC_MIN_ENTRIES


def repeated_lines_by_source(documents: Sequence[Document]) -> Dict[str, Set[str]]:
    """按文档分组找跨页重复行。

    必须按 source 分组：不同书的共同套话（「第 1 章」之类）不该被判成页眉。
    """
    per_src: Dict[str, List[Document]] = defaultdict(list)
    for doc in documents:
        per_src[doc.metadata.get("source", "?")].append(doc)

    out: Dict[str, Set[str]] = {}
    for src, docs in per_src.items():
        threshold = max(REPEAT_MIN_PAGES, int(len(docs) * REPEAT_RATIO))
        counter: Counter = Counter()
        for doc in docs:
            # 同一页里重复出现的行只算一次，否则长页会把自己的行刷成「跨页重复」
            seen = {
                ln.strip()
                for ln in doc.page_content.splitlines()
                if REPEAT_MIN_LEN <= len(ln.strip()) <= REPEAT_MAX_LEN
            }
            counter.update(seen)
        out[src] = {ln for ln, c in counter.items() if c >= threshold}
    return out


def strip_lines(text: str, drop: Set[str]) -> str:
    """删掉指定行，保留其余。不重新排版，避免改变正文结构。"""
    if not drop:
        return text
    kept = [ln for ln in text.splitlines() if ln.strip() not in drop]
    return "\n".join(kept)


def clean_documents(
    documents: Sequence[Document],
    drop_toc: bool = True,
    drop_headers: bool = True,
) -> Tuple[List[Document], Dict[str, Any]]:
    """执行过滤，返回 (清洗后的页, 统计)。

    两个开关分开，是为了做单变量对照：
    「只删页眉」和「只跳目录页」的收益来源完全不同，混在一起测就说不清是谁的功劳。
    """
    stats: Dict[str, Any] = {
        "pages_before": len(documents),
        "toc_pages_dropped": 0,
        "pages_emptied": 0,
        "header_lines_removed": 0,
        "chars_removed": 0,
    }
    docs: List[Document] = list(documents)

    if drop_toc:
        kept: List[Document] = []
        for doc in docs:
            if is_toc_page(doc.page_content):
                stats["toc_pages_dropped"] += 1
                stats["chars_removed"] += len(doc.page_content)
            else:
                kept.append(doc)
        docs = kept

    if drop_headers:
        reps = repeated_lines_by_source(docs)
        stats["repeated_lines"] = {k: sorted(v) for k, v in reps.items() if v}
        cleaned: List[Document] = []
        for doc in docs:
            drop = reps.get(doc.metadata.get("source", "?"), set())
            new_text = strip_lines(doc.page_content, drop)
            if new_text == doc.page_content:
                cleaned.append(doc)
                continue
            stats["header_lines_removed"] += (
                len(doc.page_content.splitlines()) - len(new_text.splitlines())
            )
            stats["chars_removed"] += len(doc.page_content) - len(new_text)
            if not new_text.strip():
                # 整页都是页眉页脚：删完就空了，直接丢掉这页
                stats["pages_emptied"] += 1
                continue
            cleaned.append(Document(page_content=new_text, metadata=dict(doc.metadata)))
        docs = cleaned

    stats["pages_after"] = len(docs)
    return docs, stats
