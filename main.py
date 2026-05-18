# =====================
# 1. 필요한 도구 가져오기
# =====================
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
from datetime import datetime, date
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os
import re
import joblib
import psycopg2
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

# =====================
# 2. FastAPI 서버 시작
# =====================
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
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


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# =====================
# 3. DB에서 질문 가져오기
# =====================
def get_questions_from_db():
    """
    DB chat_message 테이블에서
    USER 질문만 가져오는 함수
    실패하면 빈 리스트 반환
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT convert_from(lo_get(content), 'UTF8') AS query_text
            FROM chat_message
            WHERE sender_type = 'USER'
              AND content IS NOT NULL
            ORDER BY created_at ASC
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        questions = [row[0] for row in rows if row[0]]
        print(f"DB에서 질문 {len(questions)}개 가져왔어요")
        return questions

    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return []

# =====================
# 4. 전처리
# - 임베딩 입력: 가벼운 정제 (URL/자모/공백)
# - c-TF-IDF 토큰: Okt 명사 추출 + 명사 불용어
# =====================
_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")


def normalize_for_embedding(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _JAMO_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# 한국어에서 토픽 키워드로 의미 없는 일반 명사
NOUN_STOPWORDS = {
    "것", "거", "수", "때", "곳", "분", "게", "걸", "건",
    "년", "월", "일", "시", "번", "쪽", "중", "내",
    "정도", "경우", "관련", "이상", "이하", "이후", "이전",
}

_okt = Okt()


def korean_noun_tokenizer(text: str) -> List[str]:
    return [
        n for n in _okt.nouns(text)
        if len(n) > 1 and n not in NOUN_STOPWORDS
    ]

# =====================
# 5. 모델 설정
# =====================
embedding_model = SentenceTransformer('jhgan/ko-sroberta-sts')

umap_model = UMAP(
    n_neighbors=5,
    n_components=5,
    min_dist=0.0,
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=1,
    prediction_data=True
)

vectorizer_model = CountVectorizer(
    tokenizer=korean_noun_tokenizer,
    min_df=2,
    ngram_range=(1, 2),
)


def build_topic_model() -> BERTopic:
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=5,
        language="korean",
    )


# =====================
# 6. 체크포인트 입출력
# =====================
def save_checkpoint(model: BERTopic, qs: List[str], tps: List[int]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(
        str(TOPIC_MODEL_PATH),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,
    )
    joblib.dump({"questions": qs, "topics": tps}, METADATA_PATH)
    print(f"체크포인트 저장 완료: 질문 {len(qs)}개")


def load_checkpoint():
    if not TOPIC_MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None
    model = BERTopic.load(str(TOPIC_MODEL_PATH), embedding_model=embedding_model)
    meta = joblib.load(METADATA_PATH)
    print(f"체크포인트 로드 완료: 질문 {len(meta['questions'])}개")
    return model, meta["questions"], meta["topics"]


# =====================
# 7. 클러스터링 함수
# 매일 자정에 자동 실행
# =====================
def run_clustering():
    """매일 자정에 자동으로 호출됨"""
    global questions, topics, topic_model

    print(f"클러스터링 실행 중... {datetime.now()}")

    new_questions = get_questions_from_db()
    if not new_questions:
        print("DB에서 질문을 가져오지 못해 클러스터링을 건너뜁니다")
        return

    fresh_model = build_topic_model()
    cleaned = [normalize_for_embedding(q) for q in new_questions]
    new_topics, _ = fresh_model.fit_transform(cleaned)

    topic_model = fresh_model
    questions = new_questions
    topics = list(new_topics)

    save_checkpoint(topic_model, questions, topics)
    print(f"클러스터링 완료! {datetime.now()}")


# 서버 시작 시 초기화: 체크포인트 우선, 없으면 새로 학습
_checkpoint = load_checkpoint()
if _checkpoint is not None:
    topic_model, questions, topics = _checkpoint
else:
    questions = get_questions_from_db()
    if not questions:
        raise RuntimeError(
            "DB에서 질문을 가져올 수 없고 체크포인트도 없습니다. "
            "DB 연결을 확인하거나 체크포인트를 먼저 생성하세요."
        )
    topic_model = build_topic_model()
    _cleaned = [normalize_for_embedding(q) for q in questions]
    topics, _ = topic_model.fit_transform(_cleaned)
    topics = list(topics)
    save_checkpoint(topic_model, questions, topics)

# =====================
# 8. 자동 실행 스케줄러
# 매일 자정(00:00)에 실행
# =====================
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_clustering,
    'cron',
    hour=0,
    minute=0
)
scheduler.start()

# =====================
# 9. GET /api/trending
# 핫토픽 TOP5 반환
# =====================
@app.get("/api/trending")
def get_trending():
    """핫토픽 TOP5 반환"""
    topic_info = topic_model.get_topic_info()

    result = []
    rank = 1

    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue

        example = row["Representative_Docs"][0] if row["Representative_Docs"] else ""

        result.append({
            "rank": rank,
            "label": row["Name"],
            "example_query": example,
            "keywords": row["Representation"][:5],
            "count": row["Count"]
        })
        rank += 1

        if rank > 5:
            break

    today = date.today()

    return {
        "topics": result,
        "period": f"{today} ~ {today}",
        "updated_at": datetime.now().isoformat()
    }

# =====================
# 10. POST /api/next-actions
# 연관 질문 버튼 반환
# =====================
class NextActionRequest(BaseModel):
    query: str
    retrieved_chunk_ids: List[str] = []

@app.post("/api/next-actions")
def get_next_actions(request: NextActionRequest):
    """연관 질문 3개 반환"""
    cleaned_query = normalize_for_embedding(request.query)

    query_topic, _ = topic_model.transform([cleaned_query])
    topic_num = query_topic[0]

    related = []
    for i, t in enumerate(topics):
        if t == topic_num and questions[i] != request.query:
            related.append(questions[i])

    actions = []
    for q in related[:3]:
        actions.append({
            "label": q,
            "query": q
        })

    return {"actions": actions}

# =====================
# 11. GET /api/stats
# 일간/주간/월별 통계
# =====================
@app.get("/api/stats")
def get_stats(period: str = "daily"):
    """
    기간별 질문 통계 반환
    period: daily(일간), weekly(주간), monthly(월별)
    """
    try:
        conn = get_db_connection()
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

    except Exception as e:
        result = [
            {"date": "2026-05-14", "count": -1},
            {"date": "2026-05-15", "count": -1},
            {"date": "2026-05-16", "count": -1},
        ]

    return {
        "period": period,
        "data": result
    }

# =====================
# 12. GET /api/stats/hourly
# 시간대별 질문 수
# =====================
@app.get("/api/stats/hourly")
def get_hourly_stats():
    """시간대별 질문 수 반환"""
    try:
        conn = get_db_connection()
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

    except Exception as e:
        result = [
            {"hour": -1, "count": -1},
            {"hour": -1, "count": -1},
            {"hour": -1, "count": -1},
            {"hour": -1, "count": -1},
            {"hour": -1, "count": -1},
        ]

    return {"data": result}

# =====================
# 13. GET /api/stats/distribution
# 카테고리별 트래픽 분포
# =====================
@app.get("/api/stats/distribution")
def get_distribution():
    """카테고리별 질문 비율 반환"""
    topic_info = topic_model.get_topic_info()

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
            "percentage": percentage
        })

    return {
        "total": total,
        "data": result
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
