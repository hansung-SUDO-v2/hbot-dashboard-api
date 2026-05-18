from typing import List

import psycopg2

from apps.config import DB_CONFIG


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def fetch_user_questions() -> List[str]:
    """chat_message에서 USER 질문 본문만 가져온다. 실패 시 빈 리스트."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT convert_from(lo_get(content), 'UTF8') AS query_text
            FROM chat_message
            WHERE sender_type = 'USER'
              AND content IS NOT NULL
            ORDER BY created_at ASC
            """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        questions = [row[0] for row in rows if row[0]]
        print(f"DB에서 질문 {len(questions)}개 가져왔어요")
        return questions

    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return []
