from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.linalg import norm

from .preprocessor import TextPreprocessor


class TfIdfVectorizer:
    """Computes TF-IDF matrix and handles cosine similarity queries."""

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor
        self.corpus: List[str] = []
        self.norm_corpus: List[str] = []
        self.terms: List[str] = []
        self.tf_matrix: Optional[pd.DataFrame] = None
        self.idf_vector: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.corpus = corpus
        self.norm_corpus = [
            self.preprocessor.normalize_text(doc) for doc in self.corpus
        ]

        self._build_terms()
        self._compute_tf()
        self._compute_idf()
        return self._compute_tfidf()

    def _build_terms(self) -> None:
        words_array = [doc.split() for doc in self.norm_corpus]
        self.terms = list(set(word for words in words_array for word in words))

    def _compute_tf(self) -> None:
        features_dict = {w: 0 for w in self.terms}
        tf_data: List[List[float]] = []
        for doc in self.norm_corpus:
            bow = Counter(doc.split())
            doc_tf = {**features_dict, **bow}
            doc_length = max(len(doc.split()), 1)
            tf_data.append([doc_tf[term] / doc_length for term in self.terms])

        self.tf_matrix = pd.DataFrame(tf_data, columns=self.terms)

    def _compute_idf(self) -> None:
        N = len(self.norm_corpus)
        df = np.zeros(len(self.terms))
        for i, term in enumerate(self.terms):
            df[i] = sum(1 for doc in self.norm_corpus if term in doc.split())

        df = np.where(df == 0, 1, df)
        self.idf_vector = 1 + np.log(N / df)

    def _compute_tfidf(self) -> np.ndarray:
        if self.tf_matrix is None or self.idf_vector is None:
            raise ValueError("TF or IDF computation failed.")

        tf = np.array(self.tf_matrix, dtype="float64")
        idf = np.diag(self.idf_vector)
        tfidf = tf @ idf

        norms = norm(tfidf, axis=1)
        norms = np.where(norms == 0, 1, norms)
        self.tfidf_matrix = tfidf / norms[:, None]
        return self.tfidf_matrix

    def cosine_similarity(self, query: str) -> np.ndarray:
        if self.tfidf_matrix is None or self.idf_vector is None:
            return np.zeros(len(self.corpus))

        norm_query = self.preprocessor.normalize_text(query)
        query_bow = Counter(norm_query.split())
        query_length = max(len(norm_query.split()), 1)

        query_tf = np.array(
            [query_bow.get(term, 0) / query_length for term in self.terms]
        )
        query_tfidf = query_tf * self.idf_vector
        query_norm = norm(query_tfidf)

        if query_norm == 0:
            return np.zeros(len(self.corpus))

        query_tfidf = query_tfidf / query_norm
        similarities = np.dot(self.tfidf_matrix, query_tfidf)
        return similarities.flatten()
