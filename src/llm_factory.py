"""
LLM 工厂：把「用哪个模型」从代码里挪到配置/请求里。

**为什么要有这一层**（S7-a）：
在这之前，`ChatDeepSeek(...)` 在 5 个地方各写了一遍（main / server / 两个评测脚本 /
judge）。想换个模型就得改 5 处，而且每处都硬绑了 `DEEPSEEK_*` 三个配置项。
换模型在 RAG 里不是"顺便试试"——它同时是：
  · 成本开关（贵的 20 元/百万 vs 免费的本地 7B）
  · 判定器（换了模型等于换了裁判，历史指标不可直接比）
  · 故障切换（主通道欠费/限流时要有第二出口）
所以它必须是**运行时可选的参数**，不是编译期常量。

**为什么全部走 OpenAI 兼容协议**：
DeepSeek / 智谱 / 月之暗面 / 阿里百炼 / SiliconFlow / Ollama / vLLM
现在都提供 `POST /v1/chat/completions`。用同一个客户端 + 不同 `base_url`
就能覆盖全部，代价只有一处：**各家私有参数（如 top_p 之外的采样项）用不了**。
这个取舍对本项目有利——可替换性比单家的边缘特性值钱得多。

**免费路径有两条，别混**：
  1. `ollama` —— 真免费（自己出电费），需要本机装 Ollama 并 pull 一个模型。
     优点：数据不出机器；缺点：7B 模型的指令遵循明显弱于云端大模型，
     引用格式、拒答哨兵这些"格式要求"容易不达标。
  2. 各家的免费档（智谱 glm-4-flash、SiliconFlow 的 7B 等）—— 不要钱但有速率限制。
     **免费额度是商业策略，随时会变**，所以这里的 `free` 标记只是提示，不是承诺。

用法：
    from src.llm_factory import create_llm
    llm = create_llm()                          # 读配置，默认 provider
    llm = create_llm("ollama", "qwen2.5:7b")    # 显式指定
    llm = create_llm(role="rewrite")            # 温度按角色取默认
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 角色 -> 温度
#
# 温度是**任务属性**，不是模型属性（S6-b 踩过：一个实例承担两种任务必然把一个调错）。
# 所以这里按角色给默认值，而不是让人手工传 temperature：
#   generate 要可复现（评测要比对两次跑的结果）
#   judge    要确定（同一对（问题,块）不能今天判相关明天判不相关）
#   rewrite  要多样性（否则 N 个改写是同一句话的复读）
# ---------------------------------------------------------------------------
ROLE_TEMPERATURE: Dict[str, float] = {
    "generate": 0.0,
    "judge": 0.0,
    "rewrite": 0.7,
}


@dataclass(frozen=True)
class ModelSpec:
    """一个可选模型。`free` 只是提示，不是承诺——免费额度会变。"""

    id: str
    label: str
    free: bool = False
    note: str = ""


@dataclass(frozen=True)
class ProviderSpec:
    """
    一个 OpenAI 兼容端点。

    api_key_envs 是**候选**列表：不同人会把同一家的 key 存成不同变量名
    （`ZHIPU_API_KEY` / `GLM_API_KEY`），按顺序取第一个非空的。
    """

    name: str
    label: str
    base_url: str
    api_key_envs: Tuple[str, ...]
    models: Tuple[ModelSpec, ...]
    is_local: bool = False
    note: str = ""


# ---------------------------------------------------------------------------
# 注册表
#
# 新增一家 = 在这里加一条，别的地方一行都不用改。
# 模型清单是"常见档位"不是"全部档位"：前端允许手填模型名，
# 因为各家上新模型的速度永远快过这张表。
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        name="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com",
        api_key_envs=("LLM_API_KEY", "DEEPSEEK_API_KEY"),
        models=(
            ModelSpec("deepseek-chat", "deepseek-chat（通用，便宜）"),
            # 2026-11 实测：官方 400 报错里写的是
            # "The supported API model names are deepseek-flash, deepseek-v4-pro"。
            # 这张表就是这么被发现的 —— **模型清单最可靠的来源是报错信息本身**，
            # 别信文档、更别信记忆。deepseek-chat 仍可用（别名），故保留在首位。
            ModelSpec("deepseek-flash", "deepseek-flash（官方现名，快）"),
            ModelSpec("deepseek-v4-pro", "deepseek-v4-pro（官方现名，强）"),
            ModelSpec("deepseek-reasoner", "deepseek-reasoner（推理，贵且慢）",
                      note="若报 400 model not found，说明该名已下线"),
        ),
        note="本项目历史指标都是它跑的；换模型后旧指标不可直接比",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="Ollama（本地，真免费）",
        base_url="http://127.0.0.1:11434/v1",
        api_key_envs=("OLLAMA_API_KEY",),   # 本地不需要真 key，留空也行
        models=(
            ModelSpec("qwen2.5:7b", "qwen2.5:7b（8GB 显存可跑）", free=True),
            ModelSpec("qwen2.5:3b", "qwen2.5:3b（更快更弱）", free=True),
            ModelSpec("llama3.1:8b", "llama3.1:8b（英文强、中文一般）", free=True),
        ),
        is_local=True,
        note="不联网、不花钱，但小模型的引用格式与拒答遵从度明显更差",
    ),
    "zhipu": ProviderSpec(
        name="zhipu",
        label="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_envs=("ZHIPU_API_KEY", "GLM_API_KEY"),
        models=(
            ModelSpec("glm-4-flash", "glm-4-flash（免费档）", free=True,
                      note="免费额度以平台当期公告为准"),
            ModelSpec("glm-4-air", "glm-4-air"),
            ModelSpec("glm-4-plus", "glm-4-plus"),
        ),
    ),
    "siliconflow": ProviderSpec(
        name="siliconflow",
        label="SiliconFlow 硅基流动",
        base_url="https://api.siliconflow.cn/v1",
        api_key_envs=("SILICONFLOW_API_KEY",),
        models=(
            ModelSpec("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B（免费档）", free=True,
                      note="免费额度以平台当期公告为准"),
            ModelSpec("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5-72B"),
            ModelSpec("deepseek-ai/DeepSeek-V3", "DeepSeek-V3（托管）"),
        ),
    ),
    "qwen": ProviderSpec(
        name="qwen",
        label="阿里百炼（通义）",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_envs=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
        models=(
            ModelSpec("qwen-turbo", "qwen-turbo（便宜）"),
            ModelSpec("qwen-plus", "qwen-plus"),
            ModelSpec("qwen-max", "qwen-max"),
        ),
    ),
    "moonshot": ProviderSpec(
        name="moonshot",
        label="月之暗面 Kimi",
        base_url="https://api.moonshot.cn/v1",
        api_key_envs=("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        models=(
            ModelSpec("moonshot-v1-8k", "moonshot-v1-8k"),
            ModelSpec("moonshot-v1-32k", "moonshot-v1-32k"),
        ),
    ),
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_envs=("OPENAI_API_KEY",),
        models=(
            ModelSpec("gpt-4o-mini", "gpt-4o-mini"),
            ModelSpec("gpt-4o", "gpt-4o"),
        ),
        note="需要能直连 api.openai.com（本机通常要走代理）",
    ),
}


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def _env_key(spec: ProviderSpec) -> str:
    """按候选顺序取第一个非空的环境变量。本地端点没有 key 是合法的。"""
    for name in spec.api_key_envs:
        val = os.getenv(name, "").strip()
        if val:
            return val
    return ""


def default_provider() -> str:
    """
    未显式指定时用哪个。

    优先级：环境变量 LLM_PROVIDER > 配置里的 LLM_PROVIDER > deepseek。
    刻意不用"探测谁有 key 就用谁"：那样同一份代码在两台机器上跑的模型不同，
    指标没法比。默认值必须是**确定的**，切换必须是**显式的**。
    """
    from config import config as cfg

    return (os.getenv("LLM_PROVIDER") or getattr(cfg, "LLM_PROVIDER", "") or "deepseek").strip().lower()


def default_model(provider: str) -> str:
    # 配置里显式写了 LLM_MODEL 就用它；否则用该 provider 的第一个模型。
    # 注意：LLM_MODEL 只在「provider 也是默认 provider」时才生效，
    # 否则会出现「选了 ollama 却拿到 deepseek-chat」这种诡异组合。
    from config import config as cfg

    name = getattr(cfg, "LLM_MODEL", "") or ""
    if name and provider == default_provider():
        return name
    spec = PROVIDERS.get(provider)
    return spec.models[0].id if spec and spec.models else ""


def resolve(provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, str]:
    """
    把 (provider, model) 解析成一组确定的三元组，并做显式校验。

    校验必须显式报错而不是回落到默认值：静默回落会让「我明明选了本地免费模型，
    怎么还在扣钱」这种问题极难发现。
    """
    from config import config as cfg

    name = (provider or default_provider()).strip().lower()
    spec = PROVIDERS.get(name)
    if spec is None:
        raise ValueError(
            f"未知 provider：{name}。可选：{', '.join(sorted(PROVIDERS))}"
        )

    mdl = (model or "").strip() or default_model(name)
    if not mdl:
        raise ValueError(f"provider {name} 没有指定模型，也没有默认模型")

    base_url = spec.base_url
    # 自定义端点：允许用配置覆盖 base_url（自建 vLLM / one-api 网关 / 公司内网）
    if name == default_provider() and getattr(cfg, "LLM_API_BASE", ""):
        base_url = cfg.LLM_API_BASE

    api_key = _env_key(spec)
    if not api_key and name == default_provider() and getattr(cfg, "LLM_API_KEY", ""):
        api_key = cfg.LLM_API_KEY

    if not api_key:
        if spec.is_local:
            api_key = "local"   # OpenAI 客户端要求非空字符串，本地端点不校验
        else:
            raise ValueError(
                f"provider {name} 缺少 API key：请设置环境变量 "
                f"{' / '.join(spec.api_key_envs)}"
            )

    return {"provider": name, "model": mdl, "base_url": base_url, "api_key": api_key}


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------
def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    role: str = "generate",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    **kwargs: Any,
):
    """
    造一个可调用的 LLM。签名刻意与 `langchain_openai.ChatOpenAI` 兼容，
    这样调用方不需要知道底下是哪一家。
    """
    from langchain_openai import ChatOpenAI

    r = resolve(provider, model)
    temp = ROLE_TEMPERATURE.get(role, 0.0) if temperature is None else temperature

    params: Dict[str, Any] = {
        "model": r["model"],
        "api_key": r["api_key"],
        "base_url": r["base_url"],
        "temperature": temp,
        "streaming": streaming,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    params.update(kwargs)

    logger.info(f"LLM: provider={r['provider']} model={r['model']} role={role} temp={temp}")
    return ChatOpenAI(**params)


# ---------------------------------------------------------------------------
# 可用性（给前端下拉用）
# ---------------------------------------------------------------------------
def available_providers(probe_local: bool = True) -> List[Dict[str, Any]]:
    """
    返回所有 provider 的描述 + 是否可用。

    `available` 的判定：有 key（或本地端点端口在监听）。
    注意这只是**配置层面**的可用，不代表模型名一定对 —— 模型清单会过期，
    真正打一次才知道。所以前端还允许手填模型名。
    """
    out: List[Dict[str, Any]] = []
    for name, spec in PROVIDERS.items():
        has_key = bool(_env_key(spec))
        reachable: Optional[bool] = None
        if spec.is_local and probe_local:
            reachable = _port_open(spec.base_url)
        usable = (reachable if spec.is_local else has_key)
        out.append({
            "name": name,
            "label": spec.label,
            "base_url": spec.base_url,
            "is_local": spec.is_local,
            "has_key": has_key,
            "reachable": reachable,
            "available": bool(usable),
            "note": spec.note,
            "models": [
                {"id": m.id, "label": m.label, "free": m.free, "note": m.note}
                for m in spec.models
            ],
        })
    return out


def _port_open(url: str, timeout: float = 0.25) -> bool:
    """
    只探 TCP 端口，不发请求。

    为什么不用 HTTP GET /v1/models：服务没起时那个请求要等 DNS+连接超时，
    而这里会被前端每打开一次页面就调一次 —— 卡的是用户的等待时间。
    """
    import socket
    from urllib.parse import urlparse

    p = urlparse(url)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def describe(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """给人看的短标识，用于日志和结果文件（`deepseek:deepseek-chat`）。"""
    try:
        r = resolve(provider, model)
        return f"{r['provider']}:{r['model']}"
    except Exception:
        return f"{provider or '?'}:{model or '?'}"
