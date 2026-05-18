# =====================
# 1. 필요한 도구 가져오기
# =====================
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from datetime import datetime, date
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os
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


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# =====================
# 3. DB에서 질문 가져오기
# 실패하면 가짜 데이터 사용
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
        print("가짜 데이터로 대신 실행할게요")
        return []

# =====================
# 4. 질문 데이터
# DB 연결 실패시 가짜 데이터 사용
# =====================
questions = get_questions_from_db()

if not questions:
    questions = [
        # 장학금 관련
        "국가장학금 마감일 알려줘",
        "장학금 서류 뭐 필요해",
        "국장 2유형 얼마나 받아",
        "장학금 입금 언제 돼",
        "교내장학금 종류가 뭐야",
        "장학금 신청 자격이 뭐야",
        "국가장학금 1유형 2유형 차이",
        "장학금 취소되면 환수돼",
        "학점 몇 점이면 장학금 받아",
        "장학금 신청 기간 지났어",
        "장학금 서류 어디 내",
        "국장 신청 안 됐어",
        "장학금 조건이 뭐야",
        "등록금 장학금으로 충당돼",
        "장학금 중복 수혜 가능해",
        # 수강신청 관련
        "수강신청 몇 시부터야",
        "강의 꽉 찼는데 어떡해",
        "수강 변경 기간이 언제야",
        "전공필수 빠졌어 어떡해",
        "수강신청 오류 났어",
        "시간표 어디서 봐",
        "재수강 학점 제한 있어",
        "수강 취소하면 환불돼",
        "학점 몇 개까지 들을 수 있어",
        "수강신청 대기 어떻게 해",
        "강의 시간 겹쳐도 돼",
        "전공 수강신청 따로야",
        "수강신청 시스템 안 열려",
        "강의계획서 어디서 봐",
        "수업 결석 몇 번까지 돼",
        # 대동제 관련
        "대동제 날짜가 언제야",
        "대동제 외부인 입장 돼",
        "대동제 공연 라인업이 뭐야",
        "대동제 어디서 해",
        "대동제 입장료 있어",
        "대동제 음식 있어",
        "대동제 학생증 필요해",
        "대동제 주차 가능해",
        "대동제 비오면 취소야",
        "대동제 며칠 동안 해",
        "대동제 몇 시에 끝나",
        "대동제 티켓 어디서 사",
        # 휴학/복학 관련
        "휴학하면 등록금 돌려줘",
        "휴학 기간 최대 몇 년",
        "군휴학 신청 어떻게 해",
        "복학하면 수강신청 새로 해야 해",
        "휴학 중에 도서관 이용 돼",
        "육아휴학 신청 방법",
        "휴학하면 장학금 어떻게 돼",
        "복학 날짜가 언제야",
        "휴학 신청 기간 언제야",
        "휴학 중에 학교 올 수 있어",
        # 학사정보 관련
        "졸업 학점 몇 학점이야",
        "학사경고 몇 번이면 제적이야",
        "성적 이의신청 기간 언제야",
        "복수전공 신청 어떻게 해",
        "전과하고 싶은데",
        "졸업논문 언제 써야 해",
        "학점 포기 가능해",
        "교환학생 신청 어떻게 해",
        "부전공 신청 기간 언제야",
        "졸업 요건이 뭐야",
        "성적 열람 언제부터야",
        "학사경고 받으면 어떻게 돼",
    ]

# =====================
# 5. 불용어 처리
# 의미없는 줄임말만 제거
# =====================
stopwords = [
    "ㅎㅇ", "ㅋㅋ", "ㅠㅠ", "ㄱㅅ",
    "ㅇㅇ", "ㄴㄴ", "ㅎㅎ", "ㅜㅜ",
    "ㄷㄷ", "ㅂㅂ", "ㅈㅅ", "ㄱㄱ",
]

def remove_stopwords(text):
    """불용어 제거 함수"""
    for word in stopwords:
        text = text.replace(word, "")
    return text.strip()

# =====================
# 6. 모델 설정
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

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=5,
    language="korean"
)

# =====================
# 7. 클러스터링 함수
# 매일 자정에 자동 실행
# =====================
def run_clustering():
    """매일 자정에 자동으로 호출됨"""
    global questions, topics, probs, cleaned_questions

    print(f"클러스터링 실행 중... {datetime.now()}")

    new_questions = get_questions_from_db()
    if new_questions:
        questions = new_questions

    cleaned_questions = [remove_stopwords(q) for q in questions]
    topics, probs = topic_model.fit_transform(cleaned_questions)

    print(f"클러스터링 완료! {datetime.now()}")

# 서버 시작할 때 첫 번째 실행
cleaned_questions = [remove_stopwords(q) for q in questions]
topics, probs = topic_model.fit_transform(cleaned_questions)

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
    cleaned_query = remove_stopwords(request.query)

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
