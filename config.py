import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Загрузка перменных окружения
load_dotenv()

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent

# Устройство
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

# Qdrant
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "papers_index")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Данные
DATA_FOLDER = os.getenv("DATA_FOLDER", str(PROJECT_ROOT / "data" / "papers"))

# Модели
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")