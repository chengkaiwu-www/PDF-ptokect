"""
LLM-as-Judge：用大模型判断「召回的块是否真的能回答这个问题」。

## 为什么需要第二套判定口径

`eval/run_eval.py` 原本用「关键词字符串包含」当命中的代理判定：
    hit = any(kw in chunk_text for kw in expected_keywords)

它便宜、确定、可复现，但有一个**系统性偏差**——实测踩到了：

    q10「什么是正则化，它有什么用」  keywords = ["正则化项"]
    检索 top1 = "第 3 周-正则化(Regularization) …"   ← 明明是对的块
    判定 = MISS                                      ← 因为块里没有"正则化项"这三个字连写

这类**假阴性**（false negative）不是偶然，它服从一条规律：
    判定函数 = 字面匹配  ⟹  字面匹配强的检索方法被系统性偏袒。
    BM25 的优化目标正是字面匹配，所以关键词口径下 BM25 天然占优。
    拿这个口径去比较「向量 vs BM25 vs 混合」，等于用 BM25 的评分函数当裁判。

后果很严重：会把「指标上升」误读成「方法更好」。所以需要一套
语义判定作为对照口径 —— 这正是 LLM-as-Judge 的用途。

## 代价与坑

1. **贵且慢**：每个 (问题, 块) 对一次调用。所以必须**缓存**，
   否则同一份评测每跑一次就烧一次钱，而且不可复现（模型有随机性）。
2. **不确定性**：同一输入两次可能给不同判断。缓解手段：temperature=0 + 缓存
   + 明确的二分类 prompt（只要求"是/否"，不要打分，不要解释长文）。
3. **位置/长度偏差**：LLM 倾向于把更靠前、更长的文本判为相关。
   本项目是**逐个独立判定**（不是给一批让它排序），能规避位置偏差，
   但长度偏差仍在 —— 这是已知局限，不是"已解决"。
4. Judge 只判「这块是否包含回答所需信息」，不判「答案生成得好不好」。
   后者是另一个指标（faithfulness），S5 再做。

## 用法

    judge = LLMJudge(Path("results/judge_cache.json"))
    judge.is_relevant("什么是正则化", chunk_text)   # -> True / False
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)

# 判定 prompt 的版本号。**改 prompt 必须改它**，否则缓存 key 不变，
# 会复用旧 prompt 做出的判定，新口径看起来"没生效"。
PROMPT_VERSION = "v2-reason"

PROMPT_TEMPLATE = """你要判断一段文档材料能否为回答给定问题提供有用信息。
问题：{question}

文档材料：
{text}

判断标准：
- 只要材料中包含与问题相关的定义、公式、步骤、原因、例子中的任意一种，就算「相关」；
- 不要求材料直接给出完整答案，也不要求用词与问题一致（同义表达、术语英文/中文写法都算相关）；
- 只有材料讲的完全是另一件事时，才算「不相关」。

输出格式（严格一行，不要换行）：
  相关     -> 是 | 十个字以内的理由
  不相关   -> 否 | 十个字以内的理由

示例：
  是 | 给出了更新公式
  否 | 讲的是别的内容"""


def _parse_verdict(answer: str) -> tuple:
    """
    解析 judge 输出，返回 (是否相关, 理由)。

    容错优先：模型偶尔会多说两句（"是的，这段材料…"），
    所以只看**第一个字符**，而不是要求整句匹配。
    """
    text = (answer or "").strip()
    if not text:
        return False, "(空回复)"
    first = text[0]
    reason = ""
    if "|" in text:
        reason = text.split("|", 1)[1].strip()
    elif len(text) > 1:
        reason = text[1:].lstrip("：:，, ").strip()
    verdict = first in ("是", "Y", "y", "1") or text.lower().startswith(("yes", "true"))
    return verdict, reason[:40]


class LLMJudge:
    """
    带磁盘缓存的二分类相关性判定器。

    缓存是必需品不是优化：
      · 成本 —— 一次完整评测有数百个 (问题, 块) 对；
      · 可复现 —— 不缓存的话，同一份代码两次运行结果会因模型随机性而不同，
        这会让「指标 0.80 → 0.85」到底是真提升还是抖动变得不可分辨。
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        model: Optional[str] = None,
        max_chars: int = 1200,
    ):
        from config.config import DEEPSEEK_API_BASE, DEEPSEEK_API_KEY, DEEPSEEK_MODEL

        self.model_name = model or DEEPSEEK_MODEL
        self.api_base = DEEPSEEK_API_BASE
        self.api_key = DEEPSEEK_API_KEY
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法使用 LLM-as-Judge")

        # 截断：块长约 384 token，正常情况下不会触发；这里只是防御超长块
        self.max_chars = max_chars
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, Any] = {}
        self._llm = None
        self.calls = 0
        self.hits = 0

        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                logger.info(f"已载入 judge 缓存：{len(self.cache)} 条（{self.cache_path}）")
            except Exception as e:
                logger.warning(f"judge 缓存读取失败，忽略: {e}")

    # ------------------------------------------------------------------ 内部

    def _key(self, question: str, text: str) -> str:
        # 判定结果与「模型 + prompt 版本 + 问题 + 文档」绑定。
        # 把 prompt 版本也放进去是必须的：改了 prompt 就等于换了判定标准，
        # 复用旧缓存会让新口径看上去"没生效"，这类问题极难察觉。
        raw = f"{self.model_name}\x00{PROMPT_VERSION}\x00{question}\x00{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _ensure_llm(self):
        if self._llm is None:
            from langchain_deepseek import ChatDeepSeek

            self._llm = ChatDeepSeek(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.api_base,
                temperature=0,   # 判定任务要确定性
                max_tokens=32,   # 只要「是 | 理由」一行，限制输出让延迟和成本可控
            )
        return self._llm

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.info(f"judge 缓存已保存：{len(self.cache)} 条 -> {self.cache_path}")

    # ------------------------------------------------------------------ 对外

    def explain(self, question: str, text: str) -> tuple:
        """
        返回 (是否相关, 理由)。

        保留理由不是冗余：评测系统本身也必须**可诊断**。
        只存「是/否」时，看到 Hit@1 从 0.70 掉到 0.40 根本无法判断
        是「judge 判定过严」还是「关键词口径虚高」—— 前者要改 prompt，后者是真实结论。
        """
        key = self._key(question, text)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return bool(cached["v"]), cached.get("r", "")

        prompt = PROMPT_TEMPLATE.format(
            question=question, text=text[: self.max_chars]
        )
        try:
            resp = self._ensure_llm().invoke(prompt)
            answer = (resp.content or "").strip()
            verdict, reason = _parse_verdict(answer)
        except Exception as e:
            # 判定失败时**保守处理为不相关**，并把失败计入日志。
            # 反过来（当成相关）会让指标虚高且无人察觉 —— 静默失效更危险。
            logger.error(f"judge 调用失败，按不相关处理: {e}")
            verdict, reason = False, "(调用失败)"

        self.calls += 1
        self.cache[key] = {"v": verdict, "a": answer[:60], "r": reason}
        return verdict, reason

    def is_relevant(self, question: str, text: str) -> bool:
        """判断 text 是否能为 question 提供有用信息。"""
        return self.explain(question, text)[0]

    def stats(self) -> Dict[str, int]:
        return {"calls": self.calls, "cache_hits": self.hits, "cached_total": len(self.cache)}
