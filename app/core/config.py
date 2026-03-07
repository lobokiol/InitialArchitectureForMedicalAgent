import os
import getpass
from dotenv import load_dotenv

# Load environment variables early
load_dotenv(override=True)


def _require_env(name: str) -> str:
    """
    Fetch required environment variables.
    Falls back to an interactive prompt when running in a TTY, otherwise raises.
    """
    value = os.getenv(name)
    if value:
        return value
    if os.isatty(0):
        return getpass.getpass(f"Enter your {name}: ")
    raise RuntimeError(f"Environment variable {name} is required.")


# Core external service settings
DASHSCOPE_API_KEY: str = _require_env("DASHSCOPE_API_KEY")
ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
REDIS_URI: str = os.getenv("REDIS_URI", "redis://localhost:6379")

# Index / collection defaults
ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME", "hospital_procedures")
MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "medical_knowledge")

# Retrieval / rewrite parameters
# 从 Milvus 取出 k=15 条候选，过滤掉相似度低于 MILVUS_MIN_SIM=0.5 的，最终保留最多 MILVUS_MAX_DOCS=8 条给 LLM。
MILVUS_TOP_K: int = int(os.getenv("MILVUS_TOP_K", "15"))
MILVUS_MIN_SIM: float = float(os.getenv("MILVUS_MIN_SIM", "0.5"))
MILVUS_MAX_DOCS: int = int(os.getenv("MILVUS_MAX_DOCS", "8"))

# RRF + Rerank 参数（用于 milvus_rag 混合检索）
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "50"))  # 每路检索取 top_k
RRF_K: int = int(os.getenv("RRF_K", "60"))  # RRF 融合参数
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "10"))  # Rerank 后返回数量
MAX_REWRITE: int = int(os.getenv("MAX_REWRITE", "2"))

# Short-term history control
MAX_HISTORY_MSGS: int = int(os.getenv("MAX_HISTORY_MSGS", "12"))
TRIM_TRIGGER_MSGS: int = int(os.getenv("TRIM_TRIGGER_MSGS", "24"))

# Model defaults
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen3-max")  # qwen-turbo qwen3-max
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v2")
CHAT_BASE_URL = os.getenv(
    "CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# Misc
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))


LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000")

DIAGNOSIS_MAX_QUESTIONS: int = int(os.getenv("DIAGNOSIS_MAX_QUESTIONS", "10"))
DIAGNOSIS_SESSION_TTL: int = int(os.getenv("DIAGNOSIS_SESSION_TTL", "3600"))
