import re
from typing import List

from konlpy.tag import Okt

# 임베딩 입력 정제: URL, 자모 연속, 다중 공백 제거
_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")

# 챗봇 호출 별칭을 단일 토큰("헤이영")으로 통일.
# Okt가 "안녕 이영", "헤이 영", "이영" 등을 따로 토큰화해 토픽 라벨에
# 사람 이름처럼 노출되던 문제를 막는다. 더 긴 패턴부터 매치되도록 순서 주의.
_HEYYOUNG_RE = re.compile(r"안녕\s*이영|안녕\s*헤이영|헤이\s*영|헤이영|이영")


def normalize_for_embedding(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _JAMO_RE.sub(" ", text)
    text = _HEYYOUNG_RE.sub("헤이영", text)
    return _WS_RE.sub(" ", text).strip()


# c-TF-IDF 키워드로 의미 없는 일반 명사
NOUN_STOPWORDS = {
    "것", "거", "수", "때", "곳", "분", "게", "걸", "건",
    "년", "월", "일", "시", "번", "쪽", "중", "내",
    "정도", "경우", "관련", "이상", "이하", "이후", "이전",
}

# Okt가 "헤이영"을 ["헤", "이영"]으로 깨버리는 케이스에 대한 안전망.
# 정규식 단계의 텍스트 치환을 뚫고 토큰으로 들어오면 여기서 한 번 더 매핑한다.
TOKEN_ALIAS = {"이영": "헤이영"}

_okt = Okt()


def korean_noun_tokenizer(text: str) -> List[str]:
    out: List[str] = []
    for n in _okt.nouns(text):
        if len(n) <= 1:
            continue
        n = TOKEN_ALIAS.get(n, n)
        if n in NOUN_STOPWORDS:
            continue
        out.append(n)
    return out


# 학습 데이터에서 거를 패턴: 시스템 자동 추천 질문, 테스트성 입력
_NOISE_PATTERN_RE = re.compile(r"^\s*(?:추천\s*질문\d*|테스트.*)$")
# 같은 글자가 3회 이상 연속 반복되면 장난성 입력으로 본다.
# 예: "똥똥똥", "으하하하하", "ㅋㅋㅋ" (자모 ㅋㅋ는 normalize 단계에서 별도로 제거됨)
_REPEAT_RE = re.compile(r"(.)\1{2,}")


def is_meaningful_question(text: str) -> bool:
    """클러스터링 학습 입력으로 의미가 있을 가능성이 있는지 판단한다.

    - 5자 미만 짧은 단답 ("넵", "응", "안녕") 제외
    - 시스템 자동/테스트성 패턴 제외
    - 같은 글자 3회 이상 연속 반복("똥똥똥", "으하하하하") 제외
    - 의미 있는 명사(2자 이상)가 하나도 없는 경우 제외
    """
    s = text.strip()
    if len(s) < 5:
        return False
    if _NOISE_PATTERN_RE.match(s):
        return False
    if _REPEAT_RE.search(s):
        return False
    if not korean_noun_tokenizer(s):
        return False
    return True
