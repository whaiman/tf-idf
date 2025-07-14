# TF-IDF Text Analysis

This project implements a TF-IDF (Term Frequency-Inverse Document Frequency) model to analyze text corpora in multiple languages. It computes TF-IDF matrices and cosine similarity between queries and documents.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Language Support](#language-support)
- [Installation](#installation)
- [Usage](#usage)
- [Adding Support for Other Languages](#adding-support-for-other-languages)
- [Contributing](#contributing)
- [Dependencies](#dependencies)
- [Notes](#notes)
- [License](#license)

## Features

- Text normalization (lowercase, punctuation removal, stopword removal, lemmatization)
- Support for multiple languages with varying levels of processing (see Language Support)
- TF-IDF matrix computation
- Cosine similarity for query-document matching
- Example corpora in English, Russian, and Azerbaijani

## Project Structure

- `tfidf.py`: Contains the `TFIDF` class for text processing and TF-IDF computation.
- `main.py`: Example script demonstrating usage with sample corpora.
- `requirements.txt`: Lists required Python packages.
- `nltk_data/`: Directory for NLTK data (automatically created).
- `.gitignore`: Excludes unnecessary files (e.g., virtual environment, NLTK data).

## Language Support

The `TFIDF` class supports **any language** with the following capabilities:

- **Tokenization**: NLTK supports tokenization for many languages via `nltk.word_tokenize`. If the specified language is not supported, the English tokenizer is used as a fallback.
- **Stopwords**: NLTK provides stopword lists for numerous languages (e.g., English, Russian, French, German, Spanish, Arabic, and more). Check the full list with `nltk.corpus.stopwords.fileids()`. If stopwords are not available for a language, an empty set is used, meaning no stopwords are removed.
- **Lemmatization**:
  - **English**: Uses `WordNetLemmatizer` from NLTK.
  - **Russian**: Uses `pymorphy3` for lemmatization.
  - **Other languages**: No lemmatization is applied; words remain in their original form.

Thus, the program offers:

- **Full support** (tokenization, stopword removal, lemmatization) for English and Russian.
- **Partial support** (tokenization, stopword removal) for languages with NLTK stopword lists (e.g., French, German, Spanish, Arabic, etc.).
- **Basic support** (tokenization only) for all other languages.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/whaiman/tf-idf.git
   cd tf-idf
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. The script automatically downloads NLTK data (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) to the `nltk_data` directory on first run.

## Usage

Run the main script to process sample corpora and queries:

```bash
python main.py
```

The script processes corpora in English, Russian, and Azerbaijani, displaying:

- The corpus and queries
- Unique terms extracted
- TF-IDF matrix
- Cosine similarity scores for each query against the corpus documents

## Adding Support for Other Languages

To extend language support beyond the built-in capabilities:

1. **Custom Stopwords**:
   - Create a list of stopwords for your language and modify the `normalize_text` method in `tfidf.py` to use it. Example:

     ```python
     custom_stopwords = set(["word1", "word2"])  # Replace with stopwords for your language
     stop_words = custom_stopwords if self.language == "your_language" else set(nltk.corpus.stopwords.words(self.language))
     ```

   - Save the stopwords in a file (e.g., `stopwords_your_language.txt`) and load it dynamically if needed.

2. **Custom Lemmatization**:
   - For languages like French, Spanish, or German, integrate a library like `spacy` with appropriate language models. Example for French:

     ```bash
     pip install spacy
     python -m spacy download fr_core_news_sm
     ```

     Modify `normalize_text` in `tfidf.py`:

     ```python
     import spacy
     nlp = spacy.load("fr_core_news_sm") if self.language == "french" else None
     if self.language == "french":
         doc = nlp(text)
         filtered_tokens.append(token.lemma_ for token in doc if token.text not in stop_words)
     ```

   - For other languages, explore libraries like `stanza` or language-specific tools (e.g., `TreeTagger`).

3. **Custom Tokenization**:
   - If NLTK’s tokenizer is insufficient (e.g., for languages like Chinese or Japanese), use a specialized tokenizer. For example, with `jieba` for Chinese:

     ```bash
     pip install jieba
     ```

     Update `normalize_text`:

     ```python
     import jieba
     if self.language == "chinese":
         tokens = jieba.lcut(text)
     else:
         tokens = nltk.word_tokenize(text, language=self.language)
     ```

4. **Testing New Languages**:
   - Add a new corpus and queries for your language in `main.py`, following the structure of the existing examples (English, Azerbaijani, Russian).
   - Test the pipeline to ensure tokenization and stopword removal work as expected.

## Contributing

We welcome contributions to improve the project! Here's how you can contribute:

1. **Fork the repository:**

   - Fork the repository on GitHub and clone your fork locally:

   ```bash
   git clone https://github.com/whaiman/tf-idf.git
   cd tf-idf
   ```

2. **Create a branch:**

   - Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/your-feature-name  # For new features
   git checkout -b bugfix/your-bugfix-name   # For bug fixes
   ```

3. **Make changes:**

   - Implement your changes, such as adding support for new languages, improving performance, or fixing bugs.
   - Follow the coding style in the existing code (e.g., PEP 8 for Python).
   - Add or update tests in `main.py` if applicable.

4. **Test your changes:**

   - Run the script to ensure it works as expected:

   ```bash
   python main.py
   ```

   - Verify that your changes do not break existing functionality.

5. **Submit a pull request:**

   - Push your branch to your fork:

   ```bash
   git push origin feature/your-feature-name  # Or bugfix/your-bugfix-name
   ```

   - Open a pull request on the original repository, describing your changes in detail.

6. **Report issues:**

   - If you find bugs or have feature requests, create an issue on the GitHub repository with a clear description.

7. **Suggestions:**

   - Adding support for new languages (e.g., new stopwords or lemmatizers).
   - Optimizing the TF-IDF computation for large corpora.
   - Adding unit tests or documentation improvements.

Please ensure your contributions align with the project's MIT License.

## Dependencies

- Python 3.8+
- pandas
- numpy
- nltk
- pymorphy3 (for Russian lemmatization)

## Notes

- NLTK data is stored locally in the `nltk_data` directory for portability.
- The program avoids external file dependencies for easy sharing.
- For languages without NLTK stopword support, processing will still work but without stopword removal.

## License

MIT License
