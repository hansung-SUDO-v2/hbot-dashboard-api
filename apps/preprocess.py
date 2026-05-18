import re
from typing import List

from konlpy.tag import Okt

# 임베딩 입력 정제: URL, 자모 연속, 다중 공백 제거
_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")


def normalize_for_embedding(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _JAMO_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# c-TF-IDF 키워드로 의미 없는 일반 명사
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
