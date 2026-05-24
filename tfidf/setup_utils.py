import os

import nltk


class NLPSetup:
    """Utility class to handle NLTK dependencies."""

    @staticmethod
    def setup_nltk():
        nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)

        resources = ["punkt", "punkt_tab", "stopwords", "wordnet"]
        for resource in resources:
            try:
                nltk.data.find(f"{resource}")
            except LookupError:
                nltk.download(resource, download_dir=nltk_data_path, quiet=True)
