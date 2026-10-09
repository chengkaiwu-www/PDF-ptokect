"""
向量运算的小工具。

为什么单独成模块：L2 归一化是「索引层」和「检索层」必须**保持一致**的约定
（入库归一化、查询也必须归一化，否则分数尺度不一致）。把它集中在一处，
可以避免两边实现各自漂移——这类"两边都对但不一致"的 bug 极难排查。
"""

import hashlib
from typing import Union

import numpy as np

# 防止除零：dense embedding 理论上不会出现全零向量，但工程上必须假设输入不可信
EPS = 1e-12


def l2_normalize(
    vectors: Union[np.ndarray, list],
    eps: float = EPS,
) -> np.ndarray:
    """
    对向量做 L2 归一化（把长度缩放到 1）。支持单个向量（1-D）或一批向量（2-D）。

    为什么要归一化：
      余弦相似度 cos(a,b) = a·b / (|a||b|) 每次都要算两个范数。
      若预先让 |a| = |b| = 1，则 cos(a,b) = a·b —— 检索退化成一次内积，
      省掉每次查询都重算全库范数的开销（这正是 S3 之前代码的浪费点）。
      附带好处：内积分数落在 [-1, 1]，不同 chunk 之间可比，阈值才有意义。
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / (norm if norm > eps else eps)

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    return (arr / norms).astype(np.float32)


def content_hash(text: str) -> str:
    """
    chunk 内容指纹（sha1 前 16 位十六进制）。

    用途：
      1. 入库去重 —— 同一段文本无论来自哪个文件都只留一份；
      2. 增量索引用它判断"这块是否已经索引过"。

    为什么用内容 hash 而不是「文件 + 页 + 序号」：
      前者对「同一文件重复入库」「两个 PDF 含相同章节」都有效，
      后者只对完全相同的文件路径有效。
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path, chunk_size: int = 1 << 20) -> str:
    """文件内容指纹（sha1），用于判断 PDF 是否变动过。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()
