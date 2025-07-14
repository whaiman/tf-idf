import os
import string
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import pymorphy3
from nltk.stem import WordNetLemmatizer
from numpy.linalg import norm


class TFIDF:
    """A class to compute TF-IDF and cosine similarity for a text corpus."""

    def __init__(self, corpus, language="english"):
        """Initialize the TFIDF model with a corpus and language.

        Args:
            corpus (list): List of text documents.
            language (str): Language of the corpus (e.g., 'english', 'russian').
        """
        self.corpus = corpus
        self.language = language
        self.norm_corpus = None
        self.terms = None
        self.tf_matrix = None
        self.idf_vector = None
        self.tfidf_matrix = None
        self._setup_nltk()

    def _setup_nltk(self):
        """Set up NLTK data path and download required resources."""
        nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)

        resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
        for resource in resources:
            try:
                nltk.data.find(f"{resource}")
            except LookupError:
                nltk.download(resource, download_dir=nltk_data_path)

    def normalize_text(self, text):
        """Normalize text: remove punctuation, lowercase, remove stopwords, and lemmatize.

        Args:
            text (str): Input text to normalize.

        Returns:
            str: Normalized text.
        """
        stop_words = (
            set(nltk.corpus.stopwords.words(self.language))
            if self.language in nltk.corpus.stopwords.fileids()
            else set()
        )
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator).lower().strip()

        try:
            tokens = nltk.word_tokenize(text, language=self.language)
        except LookupError:
            tokens = nltk.word_tokenize(text, language="english")

        filtered_tokens = []
        for token in tokens:
            if token not in stop_words:
                if self.language == "russian":
                    filtered_tokens.append(
                        pymorphy3.MorphAnalyzer().parse(token)[0].normal_form
                    )
                elif self.language == "english":
                    filtered_tokens.append(WordNetLemmatizer().lemmatize(token))
                else:
                    filtered_tokens.append(token)
        return " ".join(filtered_tokens)

    def preprocess_corpus(self):
        """Apply normalization to all documents in the corpus."""
        self.norm_corpus = [self.normalize_text(doc) for doc in self.corpus]

    def build_terms(self):
        """Build a vocabulary of unique terms from the normalized corpus.

        Returns:
            list: List of unique terms.
        """
        words_array = [doc.split() for doc in self.norm_corpus]
        self.terms = list(set(word for words in words_array for word in words))
        return self.terms

    def tf(self):
        """Compute Term Frequency (TF) matrix.

        Returns:
            pd.DataFrame: TF matrix with terms as columns and documents as rows.
        """
        features_dict = {w: 0 for w in self.terms}
        tf_data = []
        for doc in self.norm_corpus:
            bow = Counter(doc.split())
            doc_tf = {**features_dict, **bow}
            tf_data.append(
                [doc_tf[term] / max(len(doc.split()), 1) for term in self.terms]
            )
        self.tf_matrix = pd.DataFrame(tf_data, columns=self.terms)
        return self.tf_matrix

    def idf(self):
        """Compute Inverse Document Frequency (IDF) vector.

        Returns:
            np.ndarray: IDF vector for each term.
        """
        N = len(self.norm_corpus)
        df = np.zeros(len(self.terms))
        for i, term in enumerate(self.terms):
            df[i] = sum(1 for doc in self.norm_corpus if term in doc.split())
        df = np.where(df == 0, 1, df)
        self.idf_vector = 1 + np.log(N / df)
        return self.idf_vector

    def tfidf(self):
        """Compute TF-IDF matrix.

        Returns:
            np.ndarray: Normalized TF-IDF matrix.
        """
        if self.tf_matrix is None:
            self.tf()
        if self.idf_vector is None:
            self.idf()
        tf = np.array(self.tf_matrix, dtype="float64")
        idf = np.diag(self.idf_vector)
        tfidf = tf @ idf
        norms = norm(tfidf, axis=1)
        norms = np.where(norms == 0, 1, norms)
        self.tfidf_matrix = tfidf / norms[:, None]
        return self.tfidf_matrix

    def cosine_similarity(self, query):
        """Compute cosine similarity between a query and the corpus documents.

        Args:
            query (str): Query text to compare with the corpus.

        Returns:
            np.ndarray: Cosine similarity scores for each document.
        """
        norm_query = self.normalize_text(query)
        query_bow = Counter(norm_query.split())
        query_tf = np.array(
            [
                query_bow.get(term, 0) / max(len(norm_query.split()), 1)
                for term in self.terms
            ]
        )
        query_tfidf = query_tf * self.idf_vector
        query_norm = norm(query_tfidf)
        if query_norm == 0:
            return np.zeros(len(self.corpus))
        query_tfidf = query_tfidf / query_norm
        similarities = np.dot(self.tfidf_matrix, query_tfidf)
        return (
            similarities.flatten()
            if self.tfidf_matrix is not None
            else np.zeros(len(self.corpus))
        )
