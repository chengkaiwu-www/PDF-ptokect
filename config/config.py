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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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
