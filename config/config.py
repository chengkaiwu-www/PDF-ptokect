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
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
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

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
