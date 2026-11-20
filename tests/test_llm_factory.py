"""
LLM provider 抽象的离线测试（不调用任何 API）。

为什么这几条值得测：
它们全是**无声失败**的候选。配错 provider 不会崩溃，只会让请求
悄悄落到另一个模型上继续花钱；判定缓存放错 key 不会报错，只会
把 A 家的判定当成 B 家的用，而缓存命中率反而变得更好看。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_factory import (  # noqa: E402
    PROVIDERS,
    ROLE_TEMPERATURE,
    available_providers,
    default_provider,
    default_model,
    describe,
    resolve,
)


def test_default_provider_is_explicit_not_guessed(monkeypatch):
    """
    默认 provider 必须来自配置，不能"谁有 key 就是谁"。

    后者会让同一份代码在两台机器上跑不同模型 —— 指标从此没法比，
    而且没有任何一行日志会告诉你为什么。
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")   # 有 key 也不该改变默认
    assert default_provider() == "deepseek"


def test_local_provider_needs_no_api_key(monkeypatch):
    """本地端点不该因为"没 key"被拒绝 —— 它本来就不需要 key。"""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    r = resolve("ollama", "qwen2.5:7b")
    assert r["api_key"] == "local"
    assert r["base_url"].startswith("http://127.0.0.1")


def test_missing_key_raises_with_actionable_message(monkeypatch):
    """
    没 key 必须报错，且错误里要写清 **该设哪个环境变量**。

    第一版是"返回 None 让调用方自己降级"，实测后果是：
    页面上选了智谱免费档，请求静默回落到 DeepSeek 继续扣钱。
    报错信息里不带变量名的话，用户还得回来读源码才能配好。
    """
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    with pytest.raises(ValueError) as ei:
        resolve("zhipu", "glm-4-flash")
    msg = str(ei.value)
    assert "ZHIPU_API_KEY" in msg


def test_unknown_provider_lists_choices():
    """写错 provider 名时要能一眼看出有哪些可选，而不是只得到一句 not found。"""
    with pytest.raises(ValueError) as ei:
        resolve("openrouter", "anything")
    msg = str(ei.value)
    for name in PROVIDERS:
        assert name in msg


def test_every_provider_has_default_model():
    """
    每个 provider 至少要有一个模型。

    否则 resolve('x') 会拿到空模型名，OpenAI 兼容端点会回一个
    "model is required" 的 400 —— 那个错误完全指不出真正的原因
    （配置表里少写了一行）。
    """
    for name, spec in PROVIDERS.items():
        assert spec.models, f"{name} 没有配任何模型"
        assert spec.base_url.startswith("http"), f"{name} 的 base_url 不合法"


def test_roles_map_to_expected_temperatures():
    """
    温度按角色给，是 S6-b 的结论：一个实例承担两种任务必然调错一个。
    生成/判定要 0（可复现），改写要高（要多样性）。
    """
    assert ROLE_TEMPERATURE["generate"] == 0
    assert ROLE_TEMPERATURE["judge"] == 0
    assert ROLE_TEMPERATURE["rewrite"] > 0.5


def test_describe_includes_provider():
    """
    人读的标识必须含 provider。

    只写模型名的话，日志里会出现两条一模一样的 "qwen2.5:7b"，
    而你分不清哪条是本地跑的（免费）、哪条是云端跑的（收费）。
    """
    label = describe("deepseek", "deepseek-chat")
    assert label == "deepseek:deepseek-chat"


def test_judge_cache_key_separates_providers(monkeypatch):
    """
    判定缓存的 key 必须区分 provider —— 这是本项目最容易"静默出错"的地方。

    两家都可能有 qwen2.5-7b：如果 key 只含模型名，B 家的判定会命中
    A 家的缓存。指标看起来正常，但那个数字是另一个模型给的。
    这里直接比对两条 key，不需要真的调模型。
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    from eval.judge import LLMJudge

    j1 = LLMJudge(cache_path=None, provider="deepseek", model="deepseek-chat")
    j2 = LLMJudge(cache_path=None, provider="ollama", model="deepseek-chat")

    # 同一个模型名、不同 provider -> key 必须不同
    assert j1._key("q", "text") != j2._key("q", "text"), \
        "不同 provider 的同名模型共用了缓存 key，会读到别人的判定"


def test_available_providers_reports_state(monkeypatch):
    """
    清单接口要能区分「没配 key」和「配了但连不上」——
    这两种情况用户要做的事完全不同（去申请 key vs 去启动服务）。
    """
    monkeypatch.setenv("ZHIPU_API_KEY", "fake")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)

    rows = {p["name"]: p for p in available_providers(probe_local=False)}
    assert rows["zhipu"]["available"] is True
    assert rows["moonshot"]["available"] is False
    assert rows["moonshot"]["has_key"] is False
