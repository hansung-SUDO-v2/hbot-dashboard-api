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

---

## 실행 방법

### 1. 패키지 설치
pip install fastapi uvicorn bertopic
pip install sentence-transformers
pip install psycopg2-binary apscheduler
pip install umap-learn hdbscan

### 2. 서버 실행
`.env`를 자동으로 읽습니다.

python main.py

또는
uvicorn main:app --reload --host 0.0.0.0 --port 8010

### 3. API 확인
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

학생 질문들
↓
불용어 처리 (줄임말 제거)
↓
ko-sroberta로 임베딩 (숫자로 변환)
↓
UMAP으로 차원 압축
↓
HDBSCAN으로 클러스터링
↓
TF-IDF로 키워드 추출
↓
TOP5 + 키워드 5개 반환

---

## 자동 실행
매일 자정(00:00)에 자동으로 클러스터링 실행
→ 새로운 질문 데이터 반영
→ 핫토픽 업데이트

---

## DB 연동
- DB: PostgreSQL
- 실제 질문 테이블: `chat_message` (`sender_type='USER'` 기준)
- 질문 본문: `convert_from(lo_get(content), 'UTF8')`
- 연결 실패시 가짜 데이터로 자동 대체

---

## 추후 개선 사항
- konlpy 추가 (한국어 형태소 분석)
- 줄임말 변환 ("국장" → "국가장학금")
- 감성 분석 추가
- 실제 DB 데이터 연동
