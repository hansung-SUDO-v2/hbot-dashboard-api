import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_CONFIG = {
    "host": os.environ["DB_HOST"],
    "port": int(os.environ["DB_PORT"]),
    "database": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
}

APP_HOST = os.environ["APP_HOST"]
APP_PORT = int(os.environ["APP_PORT"])

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOPIC_MODEL_PATH = CHECKPOINT_DIR / "topic_model"
METADATA_PATH = CHECKPOINT_DIR / "metadata.joblib"

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-sts"
