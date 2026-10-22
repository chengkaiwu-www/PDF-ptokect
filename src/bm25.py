"""
BM25 稀疏检索（Okapi BM25），零第三方依赖自研实现。

为什么不直接 `pip install rank_bm25`：
  1. BM25 的公式是本项目要讲清楚的核心知识点之一，自己写一遍才能真正解释
     「为什么需要 IDF、为什么词频要饱和、为什么要按长度归一化」；
  2. rank_bm25 不处理中文分词 —— 对中文直接按空格切会得到整段文本当"一个词"，
     BM25 彻底失效。中文 BM25 的前提是**先分词**（见 tokenize 的注释）。

公式（对一条 query q 与文档 d）：

    score(q, d) = Σ_{t ∈ q} IDF(t) · tf_sat(t, d)
                        └────┬────┘   └──────┬──────┘
                    稀有词权重更大      词频饱和 + 长度归一化

    tf_sat(t, d) = f(t,d) · (k1 + 1)
                   ─────────────────────────────────────
                   f(t,d) + k1 · (1 - b + b · |d| / avgdl)

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

三个组件各自的动机（对应三个经典问题）：
  · IDF —— 「的」这种词几乎每篇都有，匹配上它不该算证据。
            原始 RSJ 形式在 df > N/2 时 IDF 会变负（常见词命中反而扣分），
            末尾的 `+ 1` 是 Lucene 的修正，保证 IDF 恒正。
  · k1  —— 词频饱和。一个词出现 10 次并不比出现 2 次强 5 倍（边际递减），
            k1 越小饱和越快。经验值 1.2 ~ 2.0。
  · b   —— 长度归一化。长文档天然更容易命中任意词，必须惩罚。
            b=1 完全归一化，b=0 不归一化。经验值 0.75。

与向量检索的本质区别：
  向量检索比的是**语义方向**，会把罕见词/编号模糊化（"第 3.2 节" 这种 token
  在 embedding 里几乎无区分度）；BM25 要求**字面命中**，在专有名词、数字、
  API 名、代码标识符上更强。两者错得不一样，所以融合能互补 —— 这是 S4 的立足点。

性能结构：用**倒排表** postings[term][doc] = tf。
  查询只遍历 query 里的词对应的倒排链，不扫全库，复杂度 O(Σ |postings(t)|)。
  这是稀疏检索在超大语料上相对稠密检索的真正优势：罕见词的倒排链极短。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


def tokenize(text: str) -> List[str]:
    """
    把文本切成检索词。**索引期和查询期必须用同一个函数**——分词不一致
    会让同一个词在两边变成不同的 term，召回直接归零，且完全不报错。

    中文没有空格，必须分词（jieba 内置词典 + HMM 发现新词）。
    这里用 `lcut_for_search`：它在精确模式基础上把长词再切出子词
    （"中华人民共和国" → 中华人民共和国 / 华人 / 人民 / 共和 / 共和国），
    搜索场景偏召回，代价是索引词数变多。
    """
    import jieba

    tokens = jieba.lcut_for_search(text.lower())
    # 过滤纯标点 / 空白：它们没有 IDF 意义，还会凭空扩大倒排表
    return [t for t in tokens if t.strip() and not _is_punct(t)]


def _is_punct(token: str) -> bool:
    """token 全部由非字母数字字符组成时视为标点。"""
    return all(not ch.isalnum() for ch in token)


class BM25Index:
    """
    内存倒排索引 + Okapi BM25 打分。

    刻意**不做持久化**：1857 个 chunk 重建索引约 1 秒（jieba 分词是主要开销），
    落盘反而引入"索引与语料不同步"的风险。语料上到十万级再考虑缓存 tf 统计。
    """

    def __init__(
        self,
        documents: Sequence[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
        text_key: str = "page_content",
    ):
        self.k1 = k1
        self.b = b
        self.text_key = text_key
        self.n_docs = 0

        # postings[term][doc_idx] = 该 term 在 doc 中的出现次数
        self.postings: Dict[str, Dict[int, int]] = {}
        self.doc_len: List[int] = []

        self._build(documents)

    # ------------------------------------------------------------------ 建索引

    def _build(self, documents: Sequence[Dict[str, Any]]) -> None:
        total_len = 0
        for idx, doc in enumerate(documents):
            tokens = tokenize(doc.get(self.text_key, ""))
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            for term in tokens:
                chain = self.postings.get(term)
                if chain is None:
                    chain = self.postings[term] = {}
                chain[idx] = chain.get(idx, 0) + 1

        self.n_docs = len(self.doc_len)
        # avgdl：语料平均文档长度，长度归一化的基准
        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0
        logger.info(
            f"BM25 索引就绪：{self.n_docs} 篇文档，{len(self.postings)} 个 term，"
            f"avgdl={self.avgdl:.1f}"
        )

    # -------------------------------------------------------------------- 打分

    def _idf(self, term: str) -> float:
        """IDF(t) = ln((N - df + 0.5)/(df + 0.5) + 1)"""
        import math

        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        检索，返回 [(doc_idx, score), ...]，按分数降序。

        遍历的是 query 的倒排链而不是全库 —— 这是稀疏检索的速度来源。
        """
        if not self.n_docs:
            return []

        # 同一 query 里重复的词只累加一次权重（等价于 qtf 忽略）
        terms = list(dict.fromkeys(tokenize(query)))

        scores: Dict[int, float] = {}
        for term in terms:
            chain = self.postings.get(term)
            if not chain:
                continue
            idf = self._idf(term)
            for doc_idx, tf in chain.items():
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (self.k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------ 诊断

    def explain(self, query: str, doc_idx: int) -> Dict[str, float]:
        """
        拆解某条 query 对某篇文档的 BM25 得分，用于理解和调试。

        面试/学习时很有用：能看到哪个词贡献了多少分、IDF 是多少。
        """
        detail: Dict[str, float] = {}
        for term in dict.fromkeys(tokenize(query)):
            chain = self.postings.get(term, {})
            tf = chain.get(doc_idx, 0)
            if not tf:
                continue
            idf = self._idf(term)
            dl = self.doc_len[doc_idx]
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
            detail[term] = idf * tf * (self.k1 + 1.0) / denom
        return detail
