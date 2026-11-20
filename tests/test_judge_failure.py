"""
判定器失败路径的测试 —— 这条路径原本是**从没被执行过**的。

背景：`eval/judge.py` 里写了「调用失败就保守地判为不相关」，
但真出错时（DeepSeek 返回 402 余额不足）却直接崩了：
异常分支里也要写缓存，而 `answer` 只在 try 里赋值，
于是 `answer[:60]` 抛 UnboundLocalError，把整个评测打崩。

这暴露的是比一个变量更普遍的问题：**错误处理代码天然不被执行，除非你专门去执行它。**
所以补这两种情况：
  1. 配置性错误（401/402/403/404）—— 必须立刻终止，不能降级
  2. 瞬时错误（超时/限流）—— 降级成「不相关」，但要计数

跑法：D:\\anaconda\\python.exe -m pytest tests/ -q
"""

import sys
from pathlib import Path

import pytest

# 让 tests/ 能 import 到项目根目录的 eval / config / utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from judge import LLMJudge  # noqa: E402


class _FailingLLM:
    """替身：被调用就抛指定异常，并记录调用次数。"""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.n_calls = 0

    def invoke(self, prompt: str):
        self.n_calls += 1
        raise self.exc


def _judge(exc: Exception) -> tuple:
    j = LLMJudge(cache_path=None, model="fake-model")
    stub = _FailingLLM(exc)
    j._llm = stub          # 绕过真实网络：_ensure_llm() 看到非 None 就直接返回
    return j, stub


def test_insufficient_balance_aborts_instead_of_degrading():
    """402 余额不足：必须抛错终止，而不是把每道题都判成「不相关」。"""
    exc = Exception(
        "Error code: 402 - {'error': {'message': 'Insufficient Balance', "
        "'type': 'insufficient_balance_error', 'code': 'invalid_request_error'}}"
    )
    j, stub = _judge(exc)
    with pytest.raises(RuntimeError) as ei:
        j.is_relevant("什么是正则化", "第 3 周 正则化(Regularization) …")
    assert "判定不可用" in str(ei.value)
    assert stub.n_calls == 1
    # 关键：一次都不该被降级记账 —— 终止就是终止，不能留下"已处理"的假象
    assert j.failures == 0


def test_transient_error_degrades_without_crashing():
    """超时这类瞬时错误：降级为「不相关」，计数，并且不崩。"""
    j, stub = _judge(TimeoutError("Request timed out."))
    verdict, reason = j.explain("什么是正则化", "第 3 周 正则化 …")
    assert verdict is False
    assert reason == "(调用失败)"
    assert j.failures == 1
    assert stub.n_calls == 1
    # 曾经就是这里崩的：缓存里要能安全写入「没有 answer」的记录
    assert all(v["a"] == "" for v in j.cache.values())
    # 降级结果也进缓存的收益：同一题不会反复撞同样的错误重复烧调用
    j.explain("什么是正则化", "第 3 周 正则化 …")
    assert stub.n_calls == 1


def test_failure_count_is_visible_in_stats():
    """失败条数必须能从 stats() 读出来，否则报告里看不出它是被摊薄的。"""
    j, _ = _judge(ConnectionError("connection reset"))
    j.explain("q", "text")
    assert j.stats()["failures"] == 1
