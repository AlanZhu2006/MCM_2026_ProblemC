import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def hype_index(
    week_df: pd.DataFrame, alpha: float = 0.3, beta: float = 0.4, gamma: float = 0.3
) -> pd.Series:
    # Expect week_df to have 'volume', 'sentiment', 'keyword_score'
    v = week_df.get("volume", 0.0)
    s = week_df.get("sentiment", 0.0)
    k = week_df.get("keyword_score", 0.0)
    # Normalize gracefully if needed
    if week_df.shape[0] > 0:
        v_n = v / max(1.0, week_df["volume"].max())
        s_n = (s - week_df["sentiment"].min()) / max(week_df["sentiment"].ptp(), 1e-6)
        k_n = (k - week_df["keyword_score"].min()) / max(
            week_df["keyword_score"].ptp(), 1e-6
        )
        return alpha * v_n + beta * s_n + gamma * k_n
    return pd.Series([0.0] * len(week_df))


def topic_modeling(texts: list, n_topics: int = 5) -> Tuple[np.ndarray, object, object]:
    # Simple TF-IDF + LDA pipeline
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)
    topics = lda.transform(X)
    dominant = topics.argmax(axis=1)
    return dominant, lda, vectorizer


def simple_forecast(series: pd.Series, horizon: int = 4) -> np.ndarray:
    if len(series) == 0:
        return np.zeros(horizon)
    last_mean = float(series[-min(len(series), 8) :].mean())
    return np.array([last_mean] * horizon)
