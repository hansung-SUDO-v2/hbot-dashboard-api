# 한성대 AI 에이전트 RAG 대시보드 백엔드 (API)

본 프로젝트는 학생들이 질문한 데이터(Logs)를 기반으로 실시간 핫토픽을 추출하고, 통계 지표 및 연관 질문 추천 기능을 제공하는 **FastAPI 기반의 분석 대시보드 백엔드**입니다.

AI 모델 아키텍처 고도화와 한국어 전처리 파이프라인을 반영하여, 클러스터링의 정확도를 크게 개선했습니다.

---

## 주요 개선 및 반영 사항 (빌드업 포인트)

1. **데이터베이스 연결부 환경변수(.env) 격리**
   * 데이터베이스 접속 정보(`HOST`, `PASSWORD` 등)의 소스코드 하드코딩을 제거하고, `pydantic-settings`를 활용해 `.env` 파일로 안전하게 격리했습니다.
2. **한국어 형태소 분석기 (Kiwi) 도입**
   * 단순 문자열 제거 방식에서 벗어나, 형태소 분석기 `kiwipiepy`를 도입했습니다. 
   * 불필요한 조사와 어미(~했어요, ~인가요 등)를 제거하고 **핵심 명사 및 어근 위주로 전처리**를 수행하여 `BERTopic` 클러스터링 및 키워드 추출의 정확도를 대폭 끌어올렸습니다.
3. **자동 모델 갱신 스케줄러 (APScheduler)**
   * `APScheduler`를 활용하여 매일 자정(00:00)에 DB에서 새로운 질문 로그를 수집하고 AI 모델을 자동 재학습(fit_transform)하도록 구현했습니다.

---

## 시작하기 (Installation & Setup)

### 1. 필수 라이브러리 설치
프로젝트 실행을 위해 아래 패키지들을 먼저 설치해 주세요.
```bash
pip install fastapi uvicorn psycopg2-binary sentence-transformers bertopic umap-learn hdbscan apscheduler python-dotenv pydantic-settings kiwipiepy