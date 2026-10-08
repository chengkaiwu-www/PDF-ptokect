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

RETRIEVAL_TOP_K = 4

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
