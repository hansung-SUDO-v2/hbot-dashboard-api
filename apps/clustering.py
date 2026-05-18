from datetime import datetime
from typing import List

from apps import checkpoint
from apps.db import fetch_user_questions
from apps.model import build_topic_model
from apps.preprocess import is_meaningful_question, normalize_for_embedding
from apps.state import state


def _filter_for_training(raw: List[str]) -> List[str]:
    """클러스터링 학습에 의미 있는 질문만 남긴다."""
    kept = [q for q in raw if is_meaningful_question(q)]
    dropped = len(raw) - len(kept)
    if dropped:
        print(f"학습 필터: {len(raw)}개 중 {dropped}개 제외 ({len(kept)}개 사용)")
    return kept


def run_clustering() -> None:
    """매일 자정 스케줄러에서 호출. DB 새로 읽어 재학습 후 체크포인트 갱신."""
    print(f"클러스터링 실행 중... {datetime.now()}")

    raw = fetch_user_questions()
    if not raw:
        print("DB에서 질문을 가져오지 못해 클러스터링을 건너뜁니다")
        return

    new_questions = _filter_for_training(raw)
    if not new_questions:
        print("필터 후 학습 가능한 질문이 없어 클러스터링을 건너뜁니다")
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

    raw = fetch_user_questions()
    if not raw:
        raise RuntimeError(
            "DB에서 질문을 가져올 수 없고 체크포인트도 없습니다. "
            "DB 연결을 확인하거나 체크포인트를 먼저 생성하세요."
        )

    questions = _filter_for_training(raw)
    if not questions:
        raise RuntimeError(
            "필터 후 학습 가능한 질문이 없습니다. is_meaningful_question 기준을 조정하세요."
        )

    model = build_topic_model()
    cleaned = [normalize_for_embedding(q) for q in questions]
    topics, _ = model.fit_transform(cleaned)

    state.topic_model = model
    state.questions = questions
    state.topics = list(topics)

    checkpoint.save(state.topic_model, state.questions, state.topics)
