"""
父子块检索（small-to-big / parent document retrieval）。

要解决的是一个结构性的矛盾：

    块切得小 -> 向量表达聚焦、检索更准      -> 但送给 LLM 的上下文残缺，答不全
    块切得大 -> 上下文完整、答得全          -> 但一个向量要代表太多主题，检索变糊

父子块的做法是**把这两件事解耦**：用子块检索（求准），用父块喂模型（求全）。
不是折中，而是让两个阶段各自用最适合自己的粒度。

本项目的实现选择：**运行时聚合，不落盘**。
--------------------------------------------------
另一种常见做法是在建索引时就把父子关系固化（子块存向量、父块存文本，靠 parent_id 关联）。
这里没有那样做，原因是：

  1. 子块的 metadata 里已经有 (source, page)，而同一页的子块在 documents 列表里
     天然连续 —— 父块的边界信息**已经存在**，不需要再存一份；
  2. 固化父子关系就得改存储格式，并且让增量索引多一条「父块也要跟着更新」的
     维护路径，收益远小于复杂度；
  3. 动态聚合可以在检索时调窗口大小（parent_window），同一份索引能跑不同配置，
     便于做单变量对照实验 —— 而固化方案改窗口就必须重建索引。

代价（要诚实）：父块只能由**已入库的子块**拼成，边界受切分结果约束，
不是自由的语义边界。要真正的语义父块（章/节），得在切分阶段识别标题层级，
那是另一件事。

另一个必须保留的约束：**父块不跨页**。
引用校验（src/citation.py）靠 (文件名, 页码) 做集合比对，父块一旦跨页，
页码就不再是单值，引用会失真。按页聚组同时解决了这个问题。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


def _page_key(doc: Dict[str, Any]) -> Tuple[str, Any]:
    """同页判定键。不同来源即使页码相同也必须分开，否则父子块会跨文档拼接。"""
    meta = doc.get("metadata") or {}
    return (str(meta.get("source", "")), meta.get("page"))


class ParentExpander:
    """
    把命中的子块展开成同页的连续父块。

    构造时只做一次 O(n) 扫描，建立「(source,page) -> 有序 doc 下标」分组表；
    之后每次 expand 的复杂度是 O(k · 区间数)，k 为 top_k，可忽略。
    """

    def __init__(self, documents: Sequence[Dict[str, Any]], window: int = 3):
        self.documents = documents
        # 窗口 = 父块包含的子块数。1 等价于关闭（父块等于子块）。
        self.window = max(1, int(window))

        # groups: List[List[int]]，每个元素是同一页内按原始顺序排列的 doc 下标
        self.groups: List[List[int]] = []
        # doc 下标 -> (组号, 组内位置)
        self.position: Dict[int, Tuple[int, int]] = {}

        current_key: Optional[Tuple[str, Any]] = None
        current: List[int] = []
        for idx, doc in enumerate(documents):
            key = _page_key(doc)
            if key != current_key and current:
                self._close_group(current)
                current = []
            current_key = key
            current.append(idx)
        if current:
            self._close_group(current)

        logger.info(
            f"父块展开器就绪：{len(documents)} 个子块 -> {len(self.groups)} 个页分组，"
            f"窗口 {self.window}"
        )

    def _close_group(self, members: List[int]) -> None:
        gid = len(self.groups)
        self.groups.append(members)
        for pos, idx in enumerate(members):
            self.position[idx] = (gid, pos)

    # ------------------------------------------------------------------ 展开

    def expand(
        self,
        ranked: Sequence[Tuple[int, float]],
        max_parents: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            ranked: 已按相关度降序的 [(doc_idx, score)]，通常是精排后的 top_k
            max_parents: 最多返回几个父块。默认不限制 ——
                         因为父块数天然 <= 子块数，再截断反而会丢内容。

        Returns:
            父块列表，每个是与向量库 documents 同构的 dict，额外带
            parent_id / child_indices / n_children；页码等元数据取自窗口首块，
            因此 citation 校验无需改动。
        """
        if not ranked:
            return []

        intervals: List[Dict[str, Any]] = []

        for doc_idx, _score in ranked:
            info = self.position.get(doc_idx)
            if info is None:
                # 子块可能因为内容去重被移除过，跳过即可，不能报错
                continue
            gid, pos = info
            members = self.groups[gid]

            # 以命中块为中心的窗口；靠左/靠右命中时窗口整体内移，避免长度缩水
            lo = max(0, pos - self.window // 2)
            hi = min(len(members) - 1, lo + self.window - 1)

            merged = False
            for it in intervals:
                if it["gid"] != gid:
                    continue
                # 相邻或重叠即合并，否则同一页会被切成两个几乎一样的父块
                if not (hi < it["lo"] - 1 or lo > it["hi"] + 1):
                    it["lo"] = min(it["lo"], lo)
                    it["hi"] = max(it["hi"], hi)
                    merged = True
                    break
            if not merged:
                intervals.append({"gid": gid, "lo": lo, "hi": hi})
            if max_parents and len(intervals) >= max_parents:
                break

        parents: List[Dict[str, Any]] = []
        for it in intervals:
            members = self.groups[it["gid"]][it["lo"]: it["hi"] + 1]
            first = self.documents[members[0]]
            text = "\n\n".join(
                self.documents[i].get("page_content", "") for i in members
            )
            meta = dict(first.get("metadata") or {})
            meta["parent_id"] = f"{it['gid']}:{it['lo']}-{it['hi']}"
            meta["child_indices"] = list(members)
            meta["n_children"] = len(members)
            parents.append({"page_content": text, "metadata": meta})

        return parents


def describe(parents: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """给评测/前端用的父块统计。上下文长度是父子块方案真正的成本项，必须显式度量。"""
    chars = [len(p.get("page_content", "")) for p in parents]
    children = [int((p.get("metadata") or {}).get("n_children", 1)) for p in parents]
    return {
        "parents": len(parents),
        "children": sum(children),
        "chars": sum(chars),
        "avg_chars": round(sum(chars) / len(chars)) if chars else 0,
    }
