"""
Query 改写（S6-b）：在检索之前把「1 个查询」变成「多个检索请求」。

必须先想清楚的问题：为什么需要改写？
    这不是「模型不够强」，是**信息量**问题。

        "正则化怎么缓解过拟合"   —— 7 个词，查询向量里只有这 7 个词的信息

    而文档里的相关段落可能写的是「惩罚项」「权重衰减」「结构风险最小化」。
    字面上零重叠 -> BM25 直接 0 分；向量也只能靠语义相似勉强够到。
    改写的本质是：**把用户省掉、但文档里会出现的词补回来**，
    等于人工扩大了「检索入口」的覆盖面。

两种策略，机制完全不同
    Multi-Query：让模型说出同一问题的 N 种表述（换同义词 / 补全主语 / 口语转术语）
        · 扩的是「问法」的覆盖面
        · 代价：N 次检索 + 1 次 LLM 调用；风险是改写偏离原意带来噪声
    HyDE：让模型先写一段「假想答案」，**用答案去检索**
        · 依据是「表述形式不匹配」：文档库存的是陈述句，用户输入的是疑问句，
          两者在向量空间里天然有偏移；先用模型把疑问句转成陈述句再说
        · 代价：输出更长所以更贵；风险是假想答案本身错了会把检索带偏

两条设计原则（这两条比代码重要）
    1. **保留原 query，且权重高于改写**。
       改写一定会有偏差，直接「替换」等于把纠错的余地也丢了；
       保留原意为主、改写为辅，改写跑偏时顶多是噪声，不会全盘皆输。
    2. **改写失败必须退化为不改写**。
       检索是主链路，改写是增强。任何 LLM 超时/解析失败都不能让整个检索挂掉 ——
       这条决定了 `_get_variants()` 里为什么到处是 try/except 且 return []。
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)

# prompt 改了就必须换版本号：缓存 key 里含它，否则旧结果会被静默复用，
# 新 prompt「看起来没生效」——S4 在判定器上踩过一模一样的坑。
REWRITE_PROMPT_VERSION = "rw-v1"

MULTI_QUERY_PROMPT = """你在为一个中文技术文档库（机器学习、深度学习、RAG、大模型）做检索查询改写。
把下面这个问题改写成 {n} 个**语义等价但表述不同**的检索查询。

要求：
1. 每行一个查询，不要编号，不要解释，不要加引号；
2. 从不同角度改写：同义替换、补全省略的主语或宾语、口语换成术语、换一种提问方式；
3. 不得改变原问题的意图，也不得添加原问题没有的限定条件
   （例如凭空加年份、加「在 XX 框架中」这类原文没有的约束）。

用户问题：{query}

改写结果（{n} 行）："""

HYDE_PROMPT = """请就下面的问题写一段 120 字左右的说明性文字。

要求：
1. 写得像教科书或技术文档里的一段陈述句，使用该领域常见术语；
2. 不要分点、不要标题，不要写「我不确定」「以下是回答」这类话；
3. 不确定的地方可以写得笼统，但术语要正确。

问题：{query}

文字："""

# 去掉模型偶尔带上的行首标记："1. " / "1) " / "- " / "• "
_LEADING_MARK = re.compile(r"^\s*(?:[-*•·]|\d+\s*[.、)）])\s*")


def _sha1(raw: str, length: int = 20) -> str:
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def _clean_lines(text: str) -> List[str]:
    """把模型输出按行切干净：去行首编号、去空行、去引号包裹。"""
    out: List[str] = []
    for line in (text or "").splitlines():
        line = _LEADING_MARK.sub("", line).strip().strip("\"'“”「」")
        if line:
            out.append(line)
    return out


class QueryRewriter:
    """
    改写器基类：只负责三件事 —— 调 LLM、缓存、失败降级。

    子类只实现 `_build_prompt()` 和 `_extract()`，不要在这里写策略分支：
    「一种策略一个类」比「一个类里 if mode == ...」更不容易在加第三种策略时出错。
    """

    mode = "off"
    prompt_version = REWRITE_PROMPT_VERSION

    def __init__(
        self,
        llm,
        *,
        n: int = 3,
        weight: float = 0.7,
        cache_path: Optional[str] = None,
        model_name: str = "",
    ):
        """
        Args:
            llm: LangChain 风格模型（有 .invoke(prompt) 且返回带 .content 的对象）
            n: 生成几条改写（含 HyDE 时恒为 1 条假想答案）
            weight: 改写查询在 RRF 里的权重；原 query 恒为 1.0
            cache_path: 磁盘缓存路径。**评测必须给**，否则每次都重新调模型且不可复现
            model_name: 参与缓存 key，换模型不会误用旧结果
        """
        self.llm = llm
        self.n = max(1, min(int(n), 8))   # 上限 8：再多就是噪声，纯烧钱
        self.weight = float(weight)
        self.cache_path = Path(cache_path) if cache_path else None
        self.model_name = model_name or "unknown"
        self._cache: Dict[str, List[str]] = self._load_cache()
        # 计数器：用来核算「改写到底花了多少次 LLM 调用」，
        # 这是判断「这个优化值不值」的一半依据（另一半是指标）
        self.n_calls = 0
        self.n_hits = 0
        self.n_failures = 0

    # ------------------------------------------------------------ 缓存

    def _cache_key(self, query: str) -> str:
        # 影响结果的**全部**输入都要进 key：模型、prompt 版本、策略、n、问题本身。
        # 少一项就是一个静默 bug（改 prompt 后旧结果被复用）。
        raw = f"{self.model_name}|{self.prompt_version}|{self.mode}|{self.n}|{query}"
        return _sha1(raw)

    def _load_cache(self) -> Dict[str, List[str]]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as e:                                    # 缓存坏了不该影响主流程
            logger.warning(f"改写缓存读取失败，忽略：{e}")
            return {}

    def save(self) -> None:
        """把缓存落盘。评测脚本在结束时调一次即可。"""
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"改写缓存写入失败：{e}")

    # ------------------------------------------------------------ 主流程

    def expand(self, query: str) -> List[Tuple[str, float]]:
        """
        返回 [(查询文本, 权重)]。

        **第一项永远是原 query、权重 1.0** —— 这是本模块最重要的约定：
        改写只是加料，不是替换。调用方可以无条件信任 plan[0]。
        """
        plan: List[Tuple[str, float]] = [(query, 1.0)]
        seen = {query.strip()}
        for variant in self._get_variants(query):
            key = variant.strip()
            if key and key not in seen:
                seen.add(key)
                plan.append((key, self.weight))
        return plan

    def _get_variants(self, query: str) -> List[str]:
        """取改写结果：缓存 -> LLM。**任何异常都返回 []（退化为不改写）**。"""
        key = self._cache_key(query)
        if key in self._cache:
            self.n_hits += 1
            return list(self._cache[key])

        if self.llm is None:
            logger.warning("未提供 LLM，改写退化为不改写")
            return []

        try:
            prompt = self._build_prompt(query)
            self.n_calls += 1
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", None)
            if not isinstance(text, str):
                text = str(text or "")
            variants = self._extract(text)[: self.n]
            self._cache[key] = variants
            return variants
        except Exception as e:
            # 主链路（检索）不能因为增强模块挂了而失败
            self.n_failures += 1
            logger.warning(f"查询改写失败，退化为不改写：{type(e).__name__}: {e}")
            return []

    # ------------------------------------------------------------ 子类实现

    def _build_prompt(self, query: str) -> str:
        raise NotImplementedError

    def _extract(self, text: str) -> List[str]:
        raise NotImplementedError


class MultiQueryRewriter(QueryRewriter):
    """让模型说出同一问题的 N 种表述。"""

    mode = "multi_query"

    def _build_prompt(self, query: str) -> str:
        return MULTI_QUERY_PROMPT.format(n=self.n, query=query)

    def _extract(self, text: str) -> List[str]:
        return _clean_lines(text)


class HyDERewriter(QueryRewriter):
    """
    先写假设答案，再用答案去检索（Gao et al., 2022: Precise Zero-Shot Dense Retrieval）。

    这里 n 没意义（只生成一段），构造时强制为 1；
    输出可能带换行，拼成一段而不是拆成多行 —— 拆开就成了「多查询」，语义会变。
    """

    mode = "hyde"

    def __init__(self, llm, **kwargs):
        kwargs["n"] = 1
        super().__init__(llm, **kwargs)

    def _build_prompt(self, query: str) -> str:
        return HYDE_PROMPT.format(query=query)

    def _extract(self, text: str) -> List[str]:
        merged = " ".join(_clean_lines(text)).strip()
        return [merged] if merged else []


_REWRITERS = {
    "multi_query": MultiQueryRewriter,
    "hyde": HyDERewriter,
}


def create_rewriter(
    mode: Optional[str] = None,
    *,
    llm=None,
    n: Optional[int] = None,
    weight: Optional[float] = None,
    cache_path: Optional[str] = None,
    model_name: str = "",
) -> Optional[QueryRewriter]:
    """
    统一工厂。mode 为 off/none/空 时返回 None（调用方无需判断）。

    注意：**开启改写必须给 llm**。这里选择抛错而不是静默降级 ——
    「以为开了改写、其实没开」会让实验结果完全失真，这类错误很难发现。
    """
    from config.config import REWRITE_CACHE, REWRITE_N, REWRITE_WEIGHT, QUERY_REWRITE

    mode = (mode or QUERY_REWRITE).strip().lower()
    if mode in ("", "off", "none", "0", "false"):
        return None
    if mode not in _REWRITERS:
        raise ValueError(f"未知改写模式: {mode}（可选 off / multi_query / hyde）")
    if llm is None:
        raise ValueError(f"启用改写（{mode}）需要 LLM 实例，当前为 None")

    cls = _REWRITERS[mode]
    return cls(
        llm,
        n=REWRITE_N if n is None else n,
        weight=REWRITE_WEIGHT if weight is None else weight,
        cache_path=REWRITE_CACHE if cache_path is None else cache_path,
        model_name=model_name,
    )
