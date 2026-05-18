import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "rag.soay-quail.ts.net"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "hansung_agent_rag"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "1234"),
}

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8010"))

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOPIC_MODEL_PATH = CHECKPOINT_DIR / "topic_model"
METADATA_PATH = CHECKPOINT_DIR / "metadata.joblib"

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-sts"
