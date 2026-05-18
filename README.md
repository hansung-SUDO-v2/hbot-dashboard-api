# H-Bot 클러스터링 서버

## 프로젝트 소개
AI응용학과 학생회 챗봇(H-Bot)의
관리자 대시보드를 위한 클러스터링 API 서버

학생들이 챗봇에 입력한 질문들을
자동으로 분석해서
관리자가 한눈에 파악할 수 있게 해줍니다

---

## 담당
- 추천시스템팀 BE2
- 클러스터링 파이프라인 + 관리자 대시보드 API

---

## 기술 스택
- Python
- FastAPI
- BERTopic
- ko-sroberta (jhgan/ko-sroberta-sts)
- UMAP
- HDBSCAN
- KoNLPy (Okt) — 한국어 명사 토크나이저
- PostgreSQL
- APScheduler

---

## 왜 이 기술을 선택했나?

### BERTopic + ko-sroberta
- 한국어 구어체 처리 가능
- 클러스터링 + 키워드 추출 한번에 해결
- 6주 프로젝트 시간 제약에 적합

### UMAP + HDBSCAN 직접 설정
- 소규모 데이터(MVP 단계)에 맞게 최적화
- n_neighbors=5로 카테고리 분리 개선

### KoNLPy Okt + CountVectorizer
- c-TF-IDF 키워드 추출 단계에서 한국어 명사만 토큰화
- "장학금이 / 장학금을 / 장학금은"이 같은 키워드("장학금")로 묶임
- 임베딩 입력에는 영향 주지 않고 키워드 품질만 개선

---

## 프로젝트 구조

```
hansung-agent-dashboard/
├── main.py              # FastAPI 부팅 + 스케줄러 + uvicorn 진입점
├── apps/
│   ├── config.py        # 환경변수/경로 상수
│   ├── db.py            # PostgreSQL 연결 및 질문 조회
│   ├── preprocess.py    # 정규식 정제 + Okt 명사 토크나이저
│   ├── model.py         # 임베딩/UMAP/HDBSCAN/Vectorizer + BERTopic 빌더
│   ├── checkpoint.py    # 모델/메타데이터 save & load
│   ├── state.py         # 런타임 전역 상태 (topic_model, questions, topics)
│   ├── clustering.py    # initialize() + run_clustering()
│   └── routes.py        # FastAPI APIRouter (모든 API 엔드포인트)
├── checkpoints/         # 학습 결과 (자동 생성, gitignore)
├── docs/
├── requirements.txt
└── .env.example
```

---

## 실행 방법

### 0. 사전 요구사항
KoNLPy(Okt)는 내부적으로 Java VM을 띄우므로 **Java 8 이상**이 시스템에 설치되어 있어야 합니다.

```bash
# Ubuntu/Debian
sudo apt install default-jdk

# macOS
brew install openjdk
```

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수
`.env.example`을 참고해 `.env`를 작성하세요.

### 3. 서버 실행
```bash
python main.py
```
또는
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8010
```

### 4. API 확인
- 로컬: http://127.0.0.1:8010/docs
- Tailscale/LAN: http://<arm-ip>:8010/docs

---

## API 목록

| 메서드 | 주소 | 설명 |
|--------|------|------|
| GET | /api/trending | 핫토픽 TOP5 반환 |
| POST | /api/next-actions | 연관 질문 버튼 반환 |
| GET | /api/stats | 일간/주간/월별 통계 |
| GET | /api/stats/hourly | 시간대별 질문 수 |
| GET | /api/stats/distribution | 카테고리별 비율 |

---

## 클러스터링 방식

학생 질문들 (DB)
↓
임베딩 입력 정제 (URL/자모/공백 정리)
↓
ko-sroberta로 임베딩 (숫자로 변환)
↓
UMAP으로 차원 압축
↓
HDBSCAN으로 클러스터링
↓
Okt 명사 토크나이저 + c-TF-IDF로 키워드 추출
↓
TOP5 + 키워드 5개 반환

---

## 자동 실행 & 체크포인트
- 매일 자정(00:00) 자동으로 재학습 → 새 결과를 `checkpoints/`에 저장
- 서버 시작 시 `checkpoints/`에 저장본이 있으면 즉시 로드, 없을 때만 새로 학습
- 따라서 첫 부팅 외에는 무거운 `fit_transform`이 다시 일어나지 않음

---

## DB 연동
- DB: PostgreSQL
- 실제 질문 테이블: `chat_message` (`sender_type='USER'` 기준)
- 질문 본문: `convert_from(lo_get(content), 'UTF8')`
- 연결 실패 + 체크포인트 없음 → 명시적 RuntimeError 발생 (조용한 fallback 없음)

---

## 추후 개선 사항
- 줄임말 사전 (`국장` → `국가장학금` 등) 정규화 단계 추가
- 감성 분석 추가
- `representation_model` (KeyBERTInspired + MMR) 도입으로 키워드 중복 제거
- `reload=True` 분리 / 학습-서빙 프로세스 분리 (docs/ram-usage-analysis.md 참고)
