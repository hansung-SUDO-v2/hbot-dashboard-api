from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from apps.config import EMBEDDING_MODEL_NAME
from apps.preprocess import korean_noun_tokenizer

# 임베딩 모델은 무거우므로 모듈 import 시점에 한 번만 로드해 재사용한다.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_topic_model() -> BERTopic:
    umap_model = UMAP(
        n_neighbors=5,
        n_components=5,
        min_dist=0.0,
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        prediction_data=True,
    )
    # min_df=1: 작은 클러스터(20개 미만)에서도 키워드 풀세트가 살아남도록 한다.
    # ngram_range=(1, 1): bigram을 키면 "안녕 이영", "수강신청 언제" 같은
    # 띄어쓰기 포함 키워드가 라벨에 직접 박혀 가독성을 망친다. 한국어 명사는
    # 단일 어절로도 의미가 충분히 살아 unigram만 사용한다.
    vectorizer_model = CountVectorizer(
        tokenizer=korean_noun_tokenizer,
        min_df=1,
        ngram_range=(1, 1),
    )
    # MMR: 의미상 가까운 키워드 중복(예: "방법/정정/안녕")을 줄인다.
    representation_model = MaximalMarginalRelevance(diversity=0.3)
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=5,
        language="korean",
    )
