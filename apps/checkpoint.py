from typing import List, Optional, Tuple

import joblib
from bertopic import BERTopic

from apps.config import CHECKPOINT_DIR, METADATA_PATH, TOPIC_MODEL_PATH
from apps.model import embedding_model


def save(model: BERTopic, questions: List[str], topics: List[int]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(
        str(TOPIC_MODEL_PATH),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,
    )
    joblib.dump({"questions": questions, "topics": topics}, METADATA_PATH)
    print(f"체크포인트 저장 완료: 질문 {len(questions)}개")


def load() -> Optional[Tuple[BERTopic, List[str], List[int]]]:
    if not TOPIC_MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None
    model = BERTopic.load(str(TOPIC_MODEL_PATH), embedding_model=embedding_model)
    meta = joblib.load(METADATA_PATH)
    print(f"체크포인트 로드 완료: 질문 {len(meta['questions'])}개")
    return model, meta["questions"], meta["topics"]
