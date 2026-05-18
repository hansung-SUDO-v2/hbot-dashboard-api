from bertopic import BERTopic
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
    vectorizer_model = CountVectorizer(
        tokenizer=korean_noun_tokenizer,
        min_df=2,
        ngram_range=(1, 2),
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=5,
        language="korean",
    )
