from datetime import datetime

from apps import checkpoint
from apps.db import fetch_user_questions
from apps.model import build_topic_model
from apps.preprocess import normalize_for_embedding
from apps.state import state


def run_clustering() -> None:
    """매일 자정 스케줄러에서 호출. DB 새로 읽어 재학습 후 체크포인트 갱신."""
    print(f"클러스터링 실행 중... {datetime.now()}")

    new_questions = fetch_user_questions()
    if not new_questions:
        print("DB에서 질문을 가져오지 못해 클러스터링을 건너뜁니다")
        return

    fresh_model = build_topic_model()
    cleaned = [normalize_for_embedding(q) for q in new_questions]
    new_topics, _ = fresh_model.fit_transform(cleaned)

    state.topic_model = fresh_model
    state.questions = new_questions
    state.topics = list(new_topics)

    checkpoint.save(state.topic_model, state.questions, state.topics)
    print(f"클러스터링 완료! {datetime.now()}")


def initialize() -> None:
    """서버 시작 시 1회: 체크포인트 있으면 로드, 없으면 새로 학습."""
    loaded = checkpoint.load()
    if loaded is not None:
        state.topic_model, state.questions, state.topics = loaded
        return

    questions = fetch_user_questions()
    if not questions:
        raise RuntimeError(
            "DB에서 질문을 가져올 수 없고 체크포인트도 없습니다. "
            "DB 연결을 확인하거나 체크포인트를 먼저 생성하세요."
        )

    model = build_topic_model()
    cleaned = [normalize_for_embedding(q) for q in questions]
    topics, _ = model.fit_transform(cleaned)

    state.topic_model = model
    state.questions = questions
    state.topics = list(topics)

    checkpoint.save(state.topic_model, state.questions, state.topics)
