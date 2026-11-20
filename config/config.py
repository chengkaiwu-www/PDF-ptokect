import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# HuggingFace 在国内网络下通常无法直连，默认走镜像站。
# 必须在 huggingface_hub / sentence_transformers 被导入之前设置才会生效
# （它们在导入时就把 HF_ENDPOINT 读进常量），所以放在本模块顶部。
# 本项目的入口都会最先导入本模块，因此能保证时序。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
# 存储目录可用环境变量覆盖：做切分对照实验时需要并存两份索引
# （例如 CHUNK_SIZE=128 与 384 各一份），覆盖默认值比复制整个 data/ 更安全。
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(DATA_DIR / "vector_store")))
MODELS_DIR = BASE_DIR / "models"

PDF_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# --- 切分参数（S2 起支持单变量对照实验，均可用环境变量覆盖）---
# CHUNK_UNIT: "char" 按字符数 | "token" 按 embedding 模型的 token 数
#   为什么需要 token：embedding 模型的输入上限是按 token 算的（bge-small-zh 为 512）。
#   中文约 1 字 = 1 token，而英文约 3.5 字 = 1 token，
#   所以「1000 字符」对英文安全、对中文必然超限被静默截断。
# CHUNK_NORMALIZE: 是否先把 PDF 物理行合并成语义段落
# CHUNK_NORMALIZE 默认关闭：实测把 PDF 物理行合并成语义段落反而使指标下降
# （Hit@1 0.60 -> 0.55），因为小标题被并进上一段、破坏了 "\n\n" 分隔符。
# 代码保留以便复现该负面结论，详见 README「实验 6」。
CHUNK_UNIT = os.getenv("CHUNK_UNIT", "token")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "384"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
CHUNK_NORMALIZE = os.getenv("CHUNK_NORMALIZE", "0") == "1"

# --- Embedding 模型 ---
# 取值可以是 HuggingFace 模型名（如 "BAAI/bge-m3"），也可以是本地模型目录路径。
# 默认指向本地已下载的中文模型 BAAI/bge-small-zh-v1.5（512 维，约 92MB）：
#   - 替换原先的 all-MiniLM-L6-v2 —— 后者是纯英文模型，编码中文时语义空间塌陷
#   - 用本地路径加载可完全离线，避免每次启动都去 HF 检查更新
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    str(MODELS_DIR / "bge-small-zh-v1.5"),
)

# --- LLM（S7-a：从「绑死 DeepSeek」改成「可切换 provider」）---
# 所有 provider 都走 OpenAI 兼容协议，换模型 = 换 base_url + API key + 模型名。
# 注册表在 src/llm_factory.py，新增一家只改那一处。
#
# LLM_PROVIDER 的可选值见 llm_factory.PROVIDERS：
#   deepseek / ollama / zhipu / siliconflow / qwen / moonshot / openai
# 想用真免费的就选 'ollama'（需本机装 Ollama）或各家的免费档（glm-4-flash 等）。
#
# 为什么默认仍然是 deepseek：本项目 S1~S6 的全部历史指标都是它跑的，
# 换模型等于换裁判，旧指标不可直接比。默认值要稳定，切换要显式。
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

# 自定义端点覆盖（自建 vLLM / one-api 网关 / 公司内网）。
# 只在 provider == LLM_PROVIDER 时生效，避免「选了 ollama 却连到 deepseek」。
LLM_API_BASE = os.getenv("LLM_API_BASE", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()

# 旧的 DeepSeek 三项保留：已有脚本 / 环境变量仍在用，删掉会静默改行为。
# 新代码一律走 src/llm_factory.py，不要直接引用这三行。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

# --- 检索与索引（S3）---
# 后端选择：
#   'brute' —— 手写的 numpy 暴力余弦检索。精确（=真值），O(n)，用于做对照基线。
#   'faiss' —— FAISS 近似/精确索引。见下方 FAISS_* 参数。
# 保留 brute 是有意的：它是理解「ANN 到底解决了什么问题」的基准，
# 也是 benchmark 里 Recall@k 的真值来源。
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").strip().lower()

# 召回条数。
# 注意 top-k 与「相似度阈值」是两层，不要混：
#   top-k 决定「喂几块给 LLM」，阈值决定「这块够不够格」。
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# 相似度阈值：低于它的召回块被丢弃。留空 / none / off -> 不过滤（默认）。
# 为什么必须可配置而不是写死：阈值与 embedding 模型的分数分布强绑定
# （bge 的分数分布和 MiniLM 完全不同），换模型必须重新标定。
# S3 之前这里硬编码 `> 0.1`，会让召回条数莫名少于 top_k，且把
# 「检索」和「过滤」两个职责压在同一层里，无法独立调参和评估。
_thr_raw = os.getenv("RETRIEVAL_SCORE_THRESHOLD", "").strip().lower()
RETRIEVAL_SCORE_THRESHOLD = (
    float(_thr_raw) if _thr_raw and _thr_raw not in ("none", "off", "null") else None
)

# 入库时对向量做 L2 归一化。
# 归一化后 ‖v‖=1，于是 余弦相似度 == 内积，检索可省掉每次重算全库范数，
# 而且内积分数在 [-1,1] 之间可比 —— 阈值才有意义。默认开启。
INGEST_NORMALIZE = os.getenv("INGEST_NORMALIZE", "1") == "1"

# --- FAISS 参数（仅 VECTOR_BACKEND=faiss 时生效）---
#   flat : IndexFlatIP   —— 精确暴力检索，O(n)，作为 FAISS 侧的对照
#   hnsw : IndexHNSWFlat —— 分层可导航小世界图，近似检索；ef_search 可在运行时换「准/快」
#   ivf  : IndexIVFFlat  —— 倒排文件，先聚类，查询时只搜最近的 nprobe 个簇
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "hnsw").strip().lower()
FAISS_M = int(os.getenv("FAISS_M", "32"))                  # HNSW：每个节点的连接数
FAISS_EF_CONSTRUCTION = int(os.getenv("FAISS_EF_CONSTRUCTION", "200"))  # HNSW：建索引时候选池
FAISS_EF_SEARCH = int(os.getenv("FAISS_EF_SEARCH", "64"))  # HNSW：查询时候选池（越大越准越慢）
FAISS_NLIST = int(os.getenv("FAISS_NLIST", "256"))         # IVF：聚类簇数
FAISS_NPROBE = int(os.getenv("FAISS_NPROBE", "8"))         # IVF：查询时搜索的簇数

# --- 混合检索与精排（S4）---
# 召回通道：
#   'vector' —— 只用稠密向量（S3 的行为，作为基线）
#   'bm25'   —— 只用 BM25 稀疏检索
#   'hybrid' —— 两路召回 + RRF 融合（默认）
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()

# 每路召回的候选数。**必须远大于最终的 top_k**：
# 「召回宁多勿漏，精排宁准勿滥」—— 召回阶段就卡到 4 条，精排再强也没得选。
RETRIEVAL_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "50"))

# RRF 的平滑常数 k。越小，头部名次优势越大；60 是论文经验值。
# RRF 分数只用于排序，与相似度不是一个量纲，不能当阈值。
RRF_K = int(os.getenv("RRF_K", "60"))

# Cross-Encoder 精排开关。
# 关闭时（默认）走「向量+BM25+RRF」，打开后对融合候选再精排一遍。
# 实测：精排能提升 Hit@1，但每次查询要多花几十~几百毫秒（与候选数成正比）。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0") == "1"

# 精排模型。bge-reranker-base 是中文可用的最小 Cross-Encoder（约 1.1GB）。
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    str(MODELS_DIR / "bge-reranker-base"),
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Query 改写（S6-b）---
# 检索前把 1 个查询扩成多个（见 src/query_rewriter.py 的模块文档）。
#   'off'         —— 不改写（默认；改写要花 LLM 调用，不该默认开着）
#   'multi_query' —— 让 LLM 生成 N 个不同表述，各自检索后一起融合
#   'hyde'        —— 让 LLM 先写一段假想答案，用答案去检索
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "off").strip().lower()

# 生成几条改写。HyDE 恒为 1（只生成一段假想答案）。
REWRITE_N = int(os.getenv("REWRITE_N", "3"))

# 改写查询在 RRF 里的权重。原 query 恒为 1.0 —— 改写权重低于它，
# 相当于「以原意为主、改写为辅」，改写跑偏时只会变成噪声而不是主导结果。
REWRITE_WEIGHT = float(os.getenv("REWRITE_WEIGHT", "0.7"))

# 改写结果缓存。评测必须落盘，否则每次重跑都要重新调模型，且结果不可复现。
REWRITE_CACHE = str(BASE_DIR / "results" / "rewrite_cache.json")

# --- 父子块（S6-c）---
# 用子块检索（准）、用父块喂模型（全）。见 src/parent_expander.py 的模块文档。
# 默认关闭：它不提升检索指标，只改善「送给 LLM 的上下文完整性」，
# 所以它的收益只能在生成层量出来，且要同时报告上下文长度（成本）。
PARENT_EXPAND = os.getenv("PARENT_EXPAND", "0") == "1"

# 父块包含的子块数。窗口越大上下文越全、token 越贵；
# 1 等价于关闭（父块 == 子块），便于做同一条代码路径的单变量对照。
PARENT_WINDOW = int(os.getenv("PARENT_WINDOW", "3"))
