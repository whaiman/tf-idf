import re
import string
from typing import List, Optional, Set

import nltk
import pymorphy3
from nltk.stem import WordNetLemmatizer

from .setup_utils import NLPSetup


class TextPreprocessor:
    """Handles text normalization with extensible multi-language support."""

    def __init__(self, language: str = "english") -> None:
        self.language: str = language.lower()
        NLPSetup.setup_nltk()

        # Загрузка стоп-слов
        self.stop_words: Set[str] = (
            set(nltk.corpus.stopwords.words(self.language))
            if self.language in nltk.corpus.stopwords.fileids()
            else set()
        )

        self.translator = str.maketrans("", "", string.punctuation)

        self.morph_analyzer: Optional[pymorphy3.MorphAnalyzer] = None
        self.lemmatizer: Optional[WordNetLemmatizer] = None

        self._init_specific_analyzers()

    def _init_specific_analyzers(self) -> None:
        """Initialize specific tools depending on the language."""
        if self.language == "russian":
            self.morph_analyzer = pymorphy3.MorphAnalyzer()
        elif self.language == "english":
            self.lemmatizer = WordNetLemmatizer()


    def _stem_azerbaijani(self, token: str) -> str:
        """Lightweight heuristic stemmer for Azerbaijani suffixes."""
        suffixes = r"(lardan|lərdən|larındandır|lərindəndir|ların|lərin|lardan|lərdən|lar|lər|ın|in|un|ün|da|də|dan|dən|ya|yə|ı|i|u|ü)$"
        return re.sub(suffixes, "", token)

    def _lemmatize_token(self, token: str) -> str:
        """Dispatches token to the correct lemmatizer based on language."""
        if self.language == "russian" and self.morph_analyzer:
            return self.morph_analyzer.parse(token)[0].normal_form

        if self.language == "english" and self.lemmatizer:
            return self.lemmatizer.lemmatize(token)

        if self.language == "azerbaijani":
            return self._stem_azerbaijani(token)

        # Other languages ​​(e.g. Turkish, Kazakh, etc.) can be easily added here.
        # elif self.language == "turkish":
        #     return self._stem_turkish(token)

        return token  # Fallback for unsupported languages


    def normalize_text(self, text) -> str:
        """Main normalization pipeline."""
        text = text.translate(self.translator).lower().strip()

        try:
            tokens: List[str] = nltk.word_tokenize(text, language=self.language)
        except LookupError:
            tokens = nltk.word_tokenize(text, language="english")

        filtered_tokens: List[str] = []
        for token in tokens:
            if token in self.stop_words:
                continue

            # Calling the universal lemmatization manager
            lemma = self._lemmatize_token(token)
            filtered_tokens.append(lemma)

        return " ".join(filtered_tokens)
