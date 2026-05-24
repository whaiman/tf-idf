import numpy as np

from tfidf import TextPreprocessor, TfIdfVectorizer


def test_tfidf_matrix_shape():
    """We check that the matrix is ​​constructed correctly and has the correct dimensions."""

    corpus = [
        "I love eating cakes.",
        "Cakes are very tasty.",
    ]
    preprocessor = TextPreprocessor(language="english")
    vectorizer = TfIdfVectorizer(preprocessor)

    matrix = vectorizer.fit_transform(corpus)

    # 2 documents = 2 rows in the matrix
    assert matrix.shape[0] == 2
    # The term 'cake' must be extracted and lemmatized
    assert "cake" in vectorizer.terms


def test_cosine_similarity_exact_match():
    """We check that an exact match gives a cosine similarity of ~1.0."""
    corpus = ["This is a unique sentence for testing."]
    preprocessor = TextPreprocessor(language="english")
    vectorizer = TfIdfVectorizer(preprocessor)

    vectorizer.fit_transform(corpus)
    similarities = vectorizer.cosine_similarity(
        "This is a unique sentence for testing."
    )

    assert len(similarities) == 1
    # Compare with error (float math)
    assert np.isclose(similarities[0], 1.0, atol=0.01)


def test_cosine_similarity_no_match():
    """ПWe check that completely different texts give a similarity of 0.0."""
    corpus = ["Apples and oranges are fruits."]
    preprocessor = TextPreprocessor(language="english")
    vectorizer = TfIdfVectorizer(preprocessor)

    vectorizer.fit_transform(corpus)
    similarities = vectorizer.cosine_similarity("Cars and trucks drive on roads.")

    assert len(similarities) == 1
    assert np.isclose(similarities[0], 0.0)


def test_azerbaijani_stemmer_works():
    """We check that the stemmer for Azerbaijani correctly cuts off endings."""
    corpus = ["Mən alma yeməyi tortlardan daha çox sevirəm."]
    preprocessor = TextPreprocessor(language="azerbaijani")
    vectorizer = TfIdfVectorizer(preprocessor)

    vectorizer.fit_transform(corpus)

    # The word 'tortlardan' should become 'tort'
    assert "tort" in vectorizer.terms
    assert "tortlardan" not in vectorizer.terms
