from datetime import date, datetime
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from apps.db import get_connection
from apps.preprocess import normalize_for_embedding
from apps.state import state

router = APIRouter()


def _first_question_in_topic(topic_id: int) -> str:
    """학습 시점 questions/topics 매핑에서 해당 토픽의 첫 질문을 돌려준다."""
    for q, t in zip(state.questions, state.topics):
        if t == topic_id and q:
            return q
    return ""


def _clean_keywords(repr_kw, limit: int) -> list:
    """BERTopic Representation에서 빈 문자열을 걸러내 상위 N개."""
    if not isinstance(repr_kw, (list, tuple)):
        return []
    return [w for w in repr_kw if w][:limit]


@router.get("/api/trending")
def get_trending():
    """핫토픽 TOP5 반환"""
    topic_info = state.topic_model.get_topic_info()

    result = []
    rank = 1
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        # nr_topics 축소로 merged된 토픽은 Representative_Docs / Representation이
        # NaN(float)으로 들어오는 경우가 있어 list 여부를 가드한다.
        docs = row["Representative_Docs"]
        example = docs[0] if isinstance(docs, (list, tuple)) and len(docs) > 0 else ""
        if not example:
            example = _first_question_in_topic(topic_id)

        keywords = _clean_keywords(row["Representation"], 5)

        result.append({
            "rank": rank,
            "label": row["Name"],
            "example_query": example,
            "keywords": keywords,
            "count": row["Count"],
        })
        rank += 1
        if rank > 5:
            break

    today = date.today()
    return {
        "topics": result,
        "period": f"{today} ~ {today}",
        "updated_at": datetime.now().isoformat(),
    }


class NextActionRequest(BaseModel):
    query: str
    retrieved_chunk_ids: List[str] = []


@router.post("/api/next-actions")
def get_next_actions(request: NextActionRequest):
    """연관 질문 3개 반환"""
    cleaned_query = normalize_for_embedding(request.query)

    query_topic, _ = state.topic_model.transform([cleaned_query])
    topic_num = query_topic[0]

    related = [
        state.questions[i]
        for i, t in enumerate(state.topics)
        if t == topic_num and state.questions[i] != request.query
    ]

    actions = [{"label": q, "query": q} for q in related[:3]]
    return {"actions": actions}


@router.get("/api/stats")
def get_stats(period: str = "daily"):
    """기간별 질문 통계 반환. period: daily / weekly / monthly"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if period == "daily":
            cursor.execute("""
                SELECT DATE(created_at), COUNT(*)
                FROM chat_message
                WHERE sender_type = 'USER'
                  AND created_at >= NOW() - INTERVAL '1 day'
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """)
        elif period == "weekly":
            cursor.execute("""
                SELECT DATE(created_at), COUNT(*)
                FROM chat_message
                WHERE sender_type = 'USER'
                  AND created_at >= NOW() - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """)
        elif period == "monthly":
            cursor.execute("""
                SELECT DATE(created_at), COUNT(*)
                FROM chat_message
                WHERE sender_type = 'USER'
                  AND created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        result = [{"date": str(row[0]), "count": row[1]} for row in rows]

    except Exception:
        result = [
            {"date": "2026-05-14", "count": -1},
            {"date": "2026-05-15", "count": -1},
            {"date": "2026-05-16", "count": -1},
        ]

    return {"period": period, "data": result}


@router.get("/api/stats/hourly")
def get_hourly_stats():
    """시간대별 질문 수 반환"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXTRACT(HOUR FROM created_at), COUNT(*)
            FROM chat_message
            WHERE sender_type = 'USER'
              AND created_at >= NOW() - INTERVAL '7 days'
            GROUP BY EXTRACT(HOUR FROM created_at)
            ORDER BY EXTRACT(HOUR FROM created_at)
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        result = [{"hour": int(row[0]), "count": row[1]} for row in rows]

    except Exception:
        result = [{"hour": -1, "count": -1} for _ in range(5)]

    return {"data": result}


@router.get("/api/stats/distribution")
def get_distribution():
    """카테고리별 질문 비율 반환"""
    topic_info = state.topic_model.get_topic_info()

    total = sum(
        row["Count"]
        for _, row in topic_info.iterrows()
        if row["Topic"] != -1
    )

    result = []
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue

        percentage = round((row["Count"] / total) * 100, 1)
        result.append({
            "label": row["Name"],
            "count": row["Count"],
            "percentage": percentage,
        })

    return {"total": total, "data": result}


@router.get("/api/debug/keywords")
def get_all_keywords():
    """디버그용: 학습된 전체 토픽의 키워드 풀세트와 샘플 질문을 덤프한다.

    - outlier 토픽(-1) 포함
    - keywords는 BERTopic.get_topic()의 모든 단어 (trending보다 풍부)
    - sample_questions는 학습 시점 questions/topics 매핑에서 직접 추출
    """
    topic_info = state.topic_model.get_topic_info()

    topics_payload = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        terms = state.topic_model.get_topic(topic_id) or []
        keywords = [w for w, _ in terms if w] if isinstance(terms, (list, tuple)) else []

        samples = [
            state.questions[i]
            for i, t in enumerate(state.topics)
            if t == topic_id
        ][:5]

        topics_payload.append({
            "topic_id": topic_id,
            "label": row["Name"],
            "count": int(row["Count"]),
            "keywords": keywords,
            "sample_questions": samples,
        })

    return {
        "total_questions": len(state.questions),
        "topics": topics_payload,
        "updated_at": datetime.now().isoformat(),
    }
