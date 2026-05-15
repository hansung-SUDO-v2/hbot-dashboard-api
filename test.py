# =====================
# 1. 가짜 질문 데이터
# =====================
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

# 질문 수 확인
print(f"총 질문 수: {len(questions)}개")

# =====================
# 2. 불용어 처리
# =====================
# 의미없는 단어 목록
stopwords = [
    "어떻게", "언제야", "있어", "뭐야",
    "어디서", "어떡해", "됩니다", "해요",
    "있나요", "인데", "이요", "이에요",
    "어디", "언제", "어떤", "뭐",
    "좀", "그냥", "혹시", "근데",
    "ㅎㅇ", "ㅋㅋ", "ㅠㅠ", "ㄱㅅ",
]

# 불용어 제거 함수
def remove_stopwords(text):
    for word in stopwords:
        text = text.replace(word, "")
    return text.strip()

# 질문에서 불용어 제거
cleaned_questions = [remove_stopwords(q) for q in questions]

print("\n=== 불용어 처리 결과 ===")
for original, cleaned in zip(questions, cleaned_questions):
    if original != cleaned:
        print(f"전: {original}")
        print(f"후: {cleaned}")
        print()

# =====================
# 3. 모델 불러오기
# =====================
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# 한국어 임베딩 모델
embedding_model = SentenceTransformer('jhgan/ko-sroberta-sts')

# UMAP 설정
umap_model = UMAP(
    n_neighbors=5,
    n_components=5,
    min_dist=0.0,
    random_state=42
)

# HDBSCAN 설정
hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    min_samples=1,
    prediction_data=True
)

# BERTopic 설정
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=5,
    language="korean"
)

# =====================
# 4. 클러스터링 실행
# =====================
# cleaned_questions 사용 (불용어 제거된 질문)
topics, probs = topic_model.fit_transform(cleaned_questions)

# =====================
# 5. 결과 출력
# =====================
print("\n=== 클러스터링 결과 ===")
print(topic_model.get_topic_info())

print("\n=== 아웃라이어 확인 ===")
for i, topic in enumerate(topics):
    if topic == -1:
        print(f"아웃라이어: {questions[i]}")