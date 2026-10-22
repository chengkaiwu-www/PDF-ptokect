"""
Cross-Encoder 精排（rerank）。

## 与 Bi-Encoder 的根本差别

Bi-Encoder（也就是 embedding 模型）：
        q ──► encoder ──► v_q  ┐
                               ├─► cos(v_q, v_d)
        d ──► encoder ──► v_d  ┘
  · query 和 doc **各自独立**编码，互相看不到对方；
  · 所以所有 doc 的向量可以**离线预计算**，在线只算一次 query；
  · 代价：query 与 doc 之间没有任何 token 级交互，只能比"整体语义方向"。

Cross-Encoder（reranker）：
        [CLS] q [SEP] d [SEP] ──► encoder ──► 一个相关性分数
  · query 和 doc **拼成一个序列**一起过模型，每一层 self-attention 都能让
    query 的每个 token 直接看 doc 的每个 token —— 这是它更准的原因；
  · 代价：**每个候选都要跑一次完整前向**，无法预计算。1000 个候选 = 1000 次前向。

这就是工业界为什么必须「两段式」：
    Bi-Encoder 从 10⁶ 里召回 50 个（快、粗、可预计算）
    Cross-Encoder 给这 50 个精排（慢、准、候选数已被压到很小）
  把 Cross-Encoder 直接用在全集上，延迟立刻不可接受。

## 本实现

模型：BAAI/bge-reranker-base（中文/英文，XLM-RoBERTa-base 底座）。
输出是单个 logit，经 sigmoid 映射到 (0,1)，可近似当"相关概率"看
—— 但注意它不是校准过的概率，别当绝对阈值用，只用来排序。
"""

from typing import List, Optional, Sequence

from utils.logger import setup_logger

logger = setup_logger(__name__)


class CrossEncoderReranker:
    """
    懒加载的 Cross-Encoder 打分器。

    为什么懒加载：加载 reranker 要几百 MB 显存/内存 + 数秒时间。
    只做检索评测、或者配置里关掉 rerank 时，不该付这个代价。
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"加载 Cross-Encoder：{self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """
        给 (query, text) 对打分，返回 (0,1) 之间的相关度，越大越相关。

        注意输入顺序：CrossEncoder 对 [query, passage] 的顺序敏感
        （训练时就是这个顺序），不要反过来。

        ⚠️ 踩坑记录（实测踩过）：sentence-transformers 的 `CrossEncoder.predict()`
        **默认已经应用了模型自带的激活函数**（bge-reranker 是 Sigmoid），
        返回的已经是概率。如果自己再 sigmoid 一次，会发生两件坏事：
          1. 分数全被压到 0.5 附近（sigmoid(0.0001)=0.5，sigmoid(0.9955)=0.7302）；
          2. 更严重的是**排序信息被压平** —— 原本 0.0001 vs 0.0003 的差距，
             二次 sigmoid 后变成 0.50000 vs 0.50005，接近不可分。
        所以这里直接使用 predict 的返回值，不再做任何变换。
        """
        if not texts:
            return []

        import numpy as np

        model = self._ensure_model()
        pairs = [[query, t] for t in texts]
        probs = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        probs = np.asarray(probs, dtype="float32").reshape(-1)
        return probs.tolist()


def create_reranker(model_name: str, enabled: bool = True):
    """工厂：未启用时返回 None，调用方无需写 if。"""
    if not enabled:
        return None
    return CrossEncoderReranker(model_name)
