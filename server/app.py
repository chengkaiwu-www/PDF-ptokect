"""
本地 Web 服务：把 RAG pipeline 暴露成 HTTP 接口，并托管一个前端页面。

## 为什么单独一个 server/ 目录，而不是改 main.py

`main.py` 是**评测与脚本入口**（一条命令一个动作，跑完就退出）；
Web 服务是**常驻进程**，有完全不同的约束：

| | CLI | 服务 |
|---|---|---|
| 生命周期 | 启动 → 跑一次 → 退出 | 常驻，反复处理请求 |
| 模型加载 | 每次重来 20~30s，可以接受 | **只能加载一次**，否则每问一句都等半分钟 |
| BM25 索引 | 建一次用完就丢 | 每次换配置重建都是白白烧 1s |
| 精排模型 | 1.1GB，进程退出自然释放 | **必须单例复用**，否则内存爆掉 |

所以这里的核心不是「加几个路由」，而是**把昂贵的资源做成进程级单例并缓存住**。
这一点在 CLI 里完全不是问题，在服务里是第一位的。

## 三个接口

| 路由 | 作用 |
|---|---|
| `GET /api/status` | 索引与模型的真实状态（块数 / 维度 / 后端 / 语料分布）——前端据此显示徽章 |
| `POST /api/chat` | SSE 流式问答：先 `sources`，再若干 `token`，最后 `done`（含引用校验） |
| `POST /api/search` | 只检索、不调 LLM，返回每块的**双通道名次**（V# / B#）与融合分——调试台用 |

## 为什么用 SSE 而不是 WebSocket

数据是**单向**的（服务端 → 浏览器），且只需要文本。WebSocket 是双向长连接，
要多一套握手、心跳、重连逻辑，而 SSE 就是「一个不结束的 HTTP 响应」，
用 `fetch` 读流即可，**浏览器不需要任何额外协议栈**。反过来如果要做「用户中途打断
并追问」这种双向交互，WebSocket 才值得。

启动：
    set HF_HUB_OFFLINE=1
    D:\\anaconda\\python.exe server/app.py
"""

import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 本地模型已缓存，默认离线：不设这个，transformers 每次启动都会向 huggingface.co
# 查询更新，实测卡 3 分钟以上（见 README 实验 1）。设置前不要 import src.embedding。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from config.config import (  # noqa: E402
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    RERANK_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_MODE,
    RETRIEVAL_TOP_K,
    RRF_K,
    VECTOR_STORE_DIR,
)
from prompts.qa_prompt import PROMPT_VARIANTS  # noqa: E402
from src.embedding import get_embedding_model  # noqa: E402
from src.hybrid_retriever import create_hybrid_retriever  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import load_vector_store  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger(__name__)

WEB_DIR = ROOT / "web"

# ---------------------------------------------------------------------------
# 进程级单例
#
# _READY 用事件而不是「启动时同步加载」：uvicorn 在 lifespan 里同步加载会让
# 端口在 20 秒内不可连接，前端只能看到「连接被拒绝」这种没法解释的错误。
# 先把服务起起来、后台线程加载、前端轮询 status 显示「索引加载中」，体验完全不同。
# ---------------------------------------------------------------------------
_READY = threading.Event()
_STATE: Dict[str, Any] = {}
_RETRIEVERS: Dict[Tuple, Any] = {}
_RERANKER: Optional[Any] = None
_LOCK = threading.Lock()
# 精排加载单独一把锁：加载 1.1GB 要十几秒，不能让它把「只查个状态」的请求也堵住
_RERANK_LOCK = threading.Lock()


def _build_llm():
    if not DEEPSEEK_API_KEY:
        logger.warning("未配置 DEEPSEEK_API_KEY：问答接口只能返回召回片段，无法生成答案")
        return None
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        temperature=0,
        streaming=True,   # 让 llm.stream() 走真正的增量返回
    )


def _warmup() -> None:
    """后台加载 embedding 模型与向量库（约 20~30 秒，一次性）。"""
    global _STATE
    try:
        t0 = time.time()
        embedding_model = get_embedding_model()
        store = load_vector_store(VECTOR_STORE_DIR)
        if store is None:
            logger.error(f"未找到向量库：{VECTOR_STORE_DIR}，请先执行 main.py --reindex")
            _STATE = {"error": "向量库不存在，请先执行 main.py --reindex"}
            _READY.set()
            return

        llm = _build_llm()
        _STATE = {
            "embedding_model": embedding_model,
            "store": store,
            "llm": llm,
            "load_sec": round(time.time() - t0, 1),
        }
        logger.info(
            f"服务就绪：{len(store)} 块，加载耗时 {_STATE['load_sec']}s"
        )
    except Exception as exc:  # 启动失败要让前端看得见，而不是静默 500
        logger.exception("初始化失败")
        _STATE = {"error": f"初始化失败：{exc}"}
    finally:
        _READY.set()


def _state() -> Dict[str, Any]:
    if not _READY.wait(timeout=180):
        raise HTTPException(503, "服务仍在加载，请稍后重试")
    if "error" in _STATE:
        raise HTTPException(503, _STATE["error"])
    return _STATE


def _get_reranker():
    """精排模型 1.1GB —— 必须单例。工厂每构造一个检索器就会新建一个，
    对 CLI 无所谓（进程一退就回收），对常驻服务是内存灾难。"""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    with _RERANK_LOCK:
        if _RERANKER is None:
            from src.reranker import CrossEncoderReranker

            logger.info(f"加载精排模型：{RERANKER_MODEL}")
            _RERANKER = CrossEncoderReranker(RERANKER_MODEL)
            # 真正加载放在这里，只发生一次；其他线程在锁上等的是「已加载好」而不是「正在加载」
            _RERANKER.warmup()
            logger.info("精排模型就绪")
        return _RERANKER


def _get_retriever(mode: str, top_k: int, candidate_k: int, rrf_k: int, rerank: bool):
    """
    按配置缓存检索器。

    两个成本决定了必须有这个缓存：
      1. `HybridRetriever.__init__` 会**立即构建 BM25 倒排索引**（1857 块约 1s）；
      2. `create_hybrid_retriever(rerank=True)` 会新建一个 Cross-Encoder（1.1GB）。
    前端每改一次滑块就重建一次，交互会立刻变得无法忍受。

    ⚠️ 这段代码踩过一个真 bug，值得记住：
       最初是「整个函数包在 `with _LOCK:` 里，内部再调 `_get_reranker()`」，
       而 `_get_reranker()` 自己也要拿同一把 `threading.Lock` ——
       **Lock 不可重入，于是直接死锁**：请求永远不返回，日志停在「BM25 索引就绪」。
       修法有两层：
         (1) 把慢操作（加载 1.1GB 模型）挪到锁**外面** —— 持锁 10 秒等于让所有请求排队；
         (2) 双重检查：锁外先查一次，锁内再查一次，避免两个请求同时新建。
       顺带一条通用规则：**不要在锁里做 I/O 或加载模型**。
    """
    key = (mode, top_k, candidate_k, rrf_k, bool(rerank))

    cached = _RETRIEVERS.get(key)      # 锁外快查：dict 的 get 是原子操作
    if cached is not None:
        return cached

    reranker = _get_reranker() if rerank else None   # 慢加载，不持锁

    with _LOCK:
        retriever = _RETRIEVERS.get(key)             # 双重检查
        if retriever is not None:
            return retriever

        state = _state()
        t0 = time.time()
        # 先以 rerank=False 构造，避免工厂内部各自 new 一个精排模型，
        # 之后再挂上进程级单例 —— 组装逻辑仍走唯一工厂，不另写一套。
        retriever = create_hybrid_retriever(
            state["store"],
            mode=mode,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            rerank=False,
            top_k=top_k,
        )
        if reranker is not None:
            retriever.reranker = reranker
        logger.info(f"新建检索器 {key}，耗时 {round((time.time() - t0) * 1000)} ms")
        _RETRIEVERS[key] = retriever
        return retriever


# ---------------------------------------------------------------------------
# 序列化：把内部 dict 变成前端能直接用的形状
# ---------------------------------------------------------------------------
def _doc_row(doc: Dict[str, Any], rank: int, debug_row: Optional[Dict] = None) -> Dict[str, Any]:
    metadata = doc.get("metadata", {})
    source = Path(metadata.get("source", "unknown")).name
    page = metadata.get("page_label")
    if page is None:
        raw_page = metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) else "unknown"
    text = doc.get("page_content", "")
    ranks = (debug_row or {}).get("ranks") or {}
    # 只发 text 不要 preview：块文本一条就够，重复一份会让每个事件的体积翻倍，
    # 而流式响应里每一字节都要立刻过网络
    return {
        "rank": rank,
        "source": source,
        "page": page,
        "text": text,
        "vector_rank": ranks.get("vector"),
        "bm25_rank": ranks.get("bm25"),
        "rrf": (debug_row or {}).get("fused"),
        "rerank": (debug_row or {}).get("rerank"),
    }


def _retrieve_with_debug(
    question: str, mode: str, top_k: int, candidate_k: int, rrf_k: int, rerank: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    """检索并带回每块的通道名次。返回 (documents, debug, 耗时ms)。"""
    state = _state()
    retriever = _get_retriever(mode, top_k, candidate_k, rrf_k, rerank)
    t0 = time.time()
    out = retriever.retrieve(question, state["embedding_model"], top_k=top_k, return_debug=True)
    elapsed = round((time.time() - t0) * 1000, 1)
    return out["documents"], out.get("debug", {}), elapsed


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    retriever: str = RETRIEVAL_MODE
    top_k: int = Field(RETRIEVAL_TOP_K, ge=1, le=20)
    candidate_k: int = Field(RETRIEVAL_CANDIDATE_K, ge=1, le=200)
    rrf_k: int = Field(RRF_K, ge=1, le=200)
    rerank: bool = RERANK_ENABLED
    prompt_variant: str = "full"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    retriever: str = RETRIEVAL_MODE
    top_k: int = Field(5, ge=1, le=20)
    candidate_k: int = Field(RETRIEVAL_CANDIDATE_K, ge=1, le=200)
    rrf_k: int = Field(RRF_K, ge=1, le=200)
    rerank: bool = False


def _validate(req) -> None:
    if req.retriever not in ("vector", "bm25", "hybrid"):
        raise HTTPException(400, f"未知检索模式：{req.retriever}")
    if getattr(req, "prompt_variant", "full") not in PROMPT_VARIANTS:
        raise HTTPException(400, f"未知 prompt 变体：{req.prompt_variant}")


# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 只在后台线程里加载，不在 lifespan 里同步等待：
    # 否则端口在 20 秒内不可连接，前端只会看到「连接被拒绝」
    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(
    title="PDF-ptokect RAG",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


@app.get("/")
def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "web/index.html 不存在")
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.get("/api/status")
def status():
    if not _READY.is_set():
        return {"ready": False, "message": "索引与模型加载中…"}
    if "error" in _STATE:
        return {"ready": False, "error": _STATE["error"]}
    if "store" not in _STATE:
        # 加载线程尚未写入（极短的竞态窗口）
        return {"ready": False, "message": "索引与模型加载中…"}

    store = _STATE["store"]
    meta = store.meta or {}
    corpus: Dict[str, Dict[str, Any]] = {}
    for doc in store.documents:
        name = Path(doc.get("metadata", {}).get("source", "unknown")).name
        row = corpus.setdefault(name, {"source": name, "chunks": 0, "pages": set()})
        row["chunks"] += 1
        page = doc.get("metadata", {}).get("page_label")
        if page is not None:
            row["pages"].add(str(page))

    return {
        "ready": True,
        "chunks": len(store),
        "dim": meta.get("dim"),
        "backend": meta.get("backend"),
        "index_type": meta.get("index_type"),
        "normalized": meta.get("normalized"),
        "embedding_model": Path(str(meta.get("embedding_model", ""))).name,
        "load_sec": _STATE.get("load_sec"),
        "llm": DEEPSEEK_MODEL if _STATE.get("llm") else None,
        "defaults": {
            "retriever": RETRIEVAL_MODE,
            "top_k": RETRIEVAL_TOP_K,
            "candidate_k": RETRIEVAL_CANDIDATE_K,
            "rrf_k": RRF_K,
            "rerank": RERANK_ENABLED,
            "prompt_variant": "full",
        },
        "prompt_variants": list(PROMPT_VARIANTS.keys()),
        "corpus": [
            {"source": r["source"], "chunks": r["chunks"], "pages": len(r["pages"])}
            for r in sorted(corpus.values(), key=lambda x: -x["chunks"])
        ],
    }


@app.post("/api/search")
def search(req: SearchRequest):
    """只检索、不生成。返回每块的双通道名次，用于看清「这块是谁捞上来的」。"""
    _validate(req)
    documents, debug, elapsed = _retrieve_with_debug(
        req.query, req.retriever, req.top_k, req.candidate_k, req.rrf_k, req.rerank
    )
    rows = debug.get("rows") or []
    return {
        "query": req.query,
        "retrieve_ms": elapsed,
        "score_kind": debug.get("score_kind"),
        "channels": debug.get("channels"),
        "results": [
            _doc_row(doc, i + 1, rows[i] if i < len(rows) else None)
            for i, doc in enumerate(documents)
        ],
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    _validate(req)
    state = _state()

    def sse(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
            documents, debug, elapsed = _retrieve_with_debug(
                req.question, req.retriever, req.top_k, req.candidate_k, req.rrf_k, req.rerank
            )
        except Exception as exc:
            logger.exception("检索失败")
            yield sse({"type": "error", "message": f"检索失败：{exc}"})
            return

        pipeline = RAGPipeline(
            llm=state["llm"],
            retriever=_get_retriever(req.retriever, req.top_k, req.candidate_k, req.rrf_k, req.rerank),
            embedding_model=state["embedding_model"],
            prompt_variant=req.prompt_variant,
        )
        try:
            for event in pipeline.stream(req.question, documents=documents, retrieve_ms=elapsed):
                if event["type"] == "sources":
                    rows = debug.get("rows") or []
                    yield sse({
                        "type": "sources",
                        "retrieve_ms": event["retrieve_ms"],
                        "score_kind": debug.get("score_kind"),
                        "channels": debug.get("channels"),
                        "documents": [
                            _doc_row(doc, i + 1, rows[i] if i < len(rows) else None)
                            for i, doc in enumerate(event["documents"])
                        ],
                    })
                elif event["type"] == "done":
                    payload = dict(event)
                    payload["total_ms"] = round(event.get("retrieve_ms", elapsed) + event.get("gen_ms", 0), 1)
                    yield sse(payload)
                else:
                    yield sse(event)
        except Exception as exc:
            logger.exception("生成失败")
            yield sse({"type": "error", "message": f"生成失败：{exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉反向代理的缓冲，否则流式会被攒成一整块再发（本地直连无影响，
            # 但一旦放到 nginx 后面就会「不流式」了，这行是给那一天准备的）
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"打开 http://127.0.0.1:{port} 查看前端")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
