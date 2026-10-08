import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_UNIT,
    CHUNK_NORMALIZE,
    EMBEDDING_MODEL,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 中文文档的分隔符优先级：
#   段落 -> 换行 -> 中文句末标点 -> 中文次级标点 -> 空格 -> 兜底逐字
# 默认的 ["\n\n", "\n", " ", ""] 只在空格处切，而中文几乎不用空格，
# 于是切分点会落到任意位置（典型症状：把「Benefits of RAG」这类小标题
# 和它下面的内容切成两块）。
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "：", "，", " ", ""]

# 句末/段末标点：如果一行以这些符号结尾，认为语义已完整，不再与下一行合并
_SENTENCE_END = ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":", "”", "』", "」", "）", ")", "】")
# 块级起始标记：数字序号、中文序号、项目符号、标题等，遇到则另起一块
_BLOCK_START = re.compile(
    r"^\s*(?:[#>*\-•·]|\(?\d+(?:[.)、]|\s)|\d+(?:\.\d+)+|"
    r"[（(]?[一二三四五六七八九十百]+[）)、.]|第[一二三四五六七八九十百\d]+[章节讲篇部分])"
)


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef"


# 行合并的最小长度门槛：只在「上下两行都是较长正文行」时才合并。
# E2 的教训——不加长度门槛时，小标题会被并进上一段，反而毁掉了 "\n\n" 这个
# 优先级最高的分隔符，指标随之下滑（见 README「实验 5」）。
MIN_MERGE_LEN = 20


def _join_lines(prev: str, nxt: str) -> str:
    """中文之间直接拼接，拉丁文之间补一个空格（PDF 硬换行会把一个词切开）。"""
    if not prev or not nxt:
        return prev + nxt
    if _is_cjk(prev[-1]) or _is_cjk(nxt[0]):
        return prev + nxt
    if prev.endswith("-"):        # 英文换行连字符
        return prev[:-1] + nxt
    return prev + " " + nxt


def normalize_text(text: str) -> str:
    """
    把 PDF 抽出的「物理行」还原成「语义段落」。

    PDF 里的换行是排版产物（为了塞进页宽），不是语义边界。若不还原，
    下游切分会在句子中间断开，一个完整句子被拆进两个 chunk。
    规则：逐行扫描，若上一行未以句末标点结尾、且本行不是新的块级标记，则合并。

    这是启发式规则，不是万能的：对表格、代码块、多栏排版可能失效，
    所以它由 config.CHUNK_NORMALIZE 开关控制，可做 A/B 对照。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\u00a0]+", " ", text)

    out: List[str] = []
    buf = ""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if buf:
                out.append(buf)
                buf = ""
            out.append("")            # 保留段落分隔（空行是切分器的第一优先级分隔符）
            continue
        if not buf:
            buf = line
            continue
        too_short = len(buf) < MIN_MERGE_LEN or len(line) < MIN_MERGE_LEN
        if buf.endswith(_SENTENCE_END) or _BLOCK_START.match(line) or too_short:
            out.append(buf)
            buf = line
        else:
            buf = _join_lines(buf, line)
    if buf:
        out.append(buf)

    # 压缩连续空行
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


_tokenizer = None


def _get_tokenizer():
    """取 embedding 模型的 tokenizer，用于按 token 计数与超限自检。"""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _tokenizer


def _token_stats(chunks: List[Document]) -> Tuple[int, int, int]:
    tokenizer = _get_tokenizer()
    counts = [len(tokenizer.encode(c.page_content)) for c in chunks]
    if not counts:
        return 0, 0, 0
    counts.sort()
    return counts[0], counts[len(counts) // 2], counts[-1]


def check_embedding_limit(chunks: List[Document]) -> None:
    """
    切分后自检：chunk 的 token 数是否超出 embedding 模型的输入上限。

    这一步很重要——超出上限时模型不会报错，而是静默截断尾部，
    意味着「一个 chunk 的后半段内容根本没进向量库」，检索自然找不到。
    这是最隐蔽的一类 RAG 故障。
    """
    tokenizer = _get_tokenizer()
    limit = getattr(tokenizer, "model_max_length", None)
    if not limit or limit > 1_000_000:
        return
    counts = [len(tokenizer.encode(c.page_content)) for c in chunks]
    over = [n for n in counts if n > limit]
    if counts:
        counts.sort()
        logger.info(
            f"chunk token 分布: min={counts[0]} median={counts[len(counts) // 2]} max={counts[-1]} "
            f"(embedding 模型上限 {limit})"
        )
    if over:
        logger.warning(
            f"{len(over)}/{len(counts)} chunk(s) 超出 embedding 模型上限 {limit} tokens "
            f"(最大 {max(over)})，超出部分会被静默截断丢弃"
        )


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding and retrieval.

    两个可控变量（都在 config 里，便于做单变量对照实验）：
      CHUNK_UNIT      : "char" 按字符数 | "token" 按 embedding 模型 token 数
      CHUNK_NORMALIZE : 是否先把 PDF 物理行还原为语义段落
    """
    if not documents:
        logger.warning("No documents to split")
        return []

    if CHUNK_NORMALIZE:
        documents = [
            Document(page_content=normalize_text(d.page_content), metadata=dict(d.metadata))
            for d in documents
        ]

    logger.info(
        f"Splitting {len(documents)} document(s) with chunk_size={CHUNK_SIZE} "
        f"chunk_overlap={CHUNK_OVERLAP} unit={CHUNK_UNIT} normalize={CHUNK_NORMALIZE}"
    )

    if CHUNK_UNIT == "token":
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            _get_tokenizer(),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHINESE_SEPARATORS,
            add_start_index=True,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=CHINESE_SEPARATORS,
            add_start_index=True,
        )

    split_docs = splitter.split_documents(documents)
    logger.info(f"Split into {len(split_docs)} chunk(s)")
    check_embedding_limit(split_docs)

    return split_docs
