# H-Bot Dashboard RAM Usage Analysis

작성일: 2026-05-16 (Asia/Seoul)

## 요약

현재 구조에서 메모리 사용량은 꽤 큰 편입니다.

- 클러스터링/초기 로딩 1회 기준 peak RSS: **약 1.38 GB**
- 서버 실행 상태(`reload=True`) 기준 총 사용량: **약 3.0 GB 수준**
- 원인: **모델 로딩 + BERTopic fit_transform + reload로 인한 중복 프로세스**

---

## 측정 결과

### 1. 클러스터링(초기 import + fit_transform) 시점

다음 과정을 포함한 새 프로세스 기준:

- DB 질문 로드
- `SentenceTransformer` 로드
- `BERTopic` 초기화
- `topic_model.fit_transform(cleaned_questions)` 실행

측정값:

- `maxrss_kb = 1448308`
- 대략 **1.38 GB**

---

### 2. 서버 실행 중 메모리 사용량

측정 당시 실행 프로세스:

- 부모: `python main.py`
- 자식: multiprocessing worker/reload process

#### 부모 프로세스

- RSS: `1617200 kB` ≈ **1.54 GB**
- PSS: `1507312 kB` ≈ **1.44 GB**
- HWM: `1997768 kB` ≈ **1.91 GB**

#### 자식 프로세스

- RSS: `1665832 kB` ≈ **1.59 GB**
- PSS: `1555901 kB` ≈ **1.48 GB**
- HWM: `1997404 kB` ≈ **1.90 GB**

#### 합산

- 총 RSS: 약 **3.13 GB**
- 총 PSS: 약 **3.0 GB**

> 참고: RSS는 단순 합산 시 공유 메모리까지 중복 계산될 수 있으므로, 실제 체감 점유는 PSS가 더 현실적입니다.

---

## 왜 많이 먹는가

### 1. `reload=True`

현재 `main.py` 마지막 실행부:

```python
uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
```

이 설정 때문에:

- 부모 프로세스 1개
- 자식 프로세스 1개

가 함께 떠 있고, 둘 다 모델/클러스터링 상태를 사실상 들고 있어 메모리 사용량이 크게 증가합니다.

### 2. import 시점에 바로 학습 수행

현재 구조는 모듈 import 시점에 바로 아래 작업을 수행합니다.

- 질문 로드
- 전처리
- `fit_transform()`

즉 서버 기동 자체가 곧 대형 메모리 작업입니다.

### 3. 모델 자체 비용

다음 조합이 메모리를 사용합니다.

- `SentenceTransformer('jhgan/ko-sroberta-sts')`
- `BERTopic`
- `UMAP`
- `HDBSCAN`
- 질문 임베딩/클러스터링 결과

---

## 운영 관점 해석

### 개발 모드

`reload=True`는 편하지만 메모리 소모가 큽니다.

### 상시 실행/운영 모드

`reload=False`로 두는 편이 적절합니다.

예상 효과:

- 상주 메모리를 대략 **1.5~1.7 GB 수준**으로 낮출 가능성이 큼

---

## 권장 개선안

### 1. dev/prod 실행 분리

- 개발: `reload=True`
- 운영: `reload=False`

### 2. 클러스터링 결과 체크포인트 저장

현재는 체크포인트가 없어 재시작 시 다시 계산합니다.

권장:

- `topic_model.save(...)`
- 메타데이터(JSON/joblib/pickle) 저장
- 서버 시작 시 저장본이 있으면 우선 로드

### 3. 학습과 API 프로세스 분리

추천 구조:

- 배치/스케줄러 프로세스: 클러스터링 수행 및 저장
- API 프로세스: 저장된 결과만 읽어서 제공

이렇게 하면 API 서버 상주 메모리를 더 줄일 수 있습니다.

### 4. import 시점 학습 제거

현재처럼 import 때 바로 `fit_transform()` 하지 말고:

- 명시적 초기화 함수로 분리하거나
- 캐시가 없을 때만 수행

하는 방식이 더 안전합니다.

---

## 결론

현재 구조 기준:

- **학습/초기 클러스터링 1회:** 약 **1.38 GB**
- **실행 중(`reload=True`) 총 메모리:** 약 **3.0 GB**

따라서 이 프로젝트는 작은 유틸 API 치고는 메모리 요구량이 꽤 높은 편이며,
가장 먼저 할 최적화는 **`reload` 분리**, **체크포인트 저장**, **학습/서빙 분리**입니다.
