from tfidf import TFIDF


def main():
    """Run TF-IDF analysis on sample corpora in multiple languages."""
    # Corpora for different languages
    english_corpus = [
        "I love eating cakes more than apples.",
        "Oranges are better than pies in summer.",
        "Apple orchards bloom near the road.",
        "He traveled across the river to the market.",
    ]
    english_queries = [
        "Cakes are made with flour, oranges, and water.",
        "Pies disappeared where I arrived.",
        "He carried a cake across the river.",
    ]
    azerbaijani_corpus = [
        "Mən alma yeməyi tortlardan daha çox sevirəm.",
        "Yayda portağallar piroqlardan daha yaxşıdır.",
        "Alma bağları yolun yaxınlığında çiçəklənir.",
        "O, bazar üçün çayın o tayına səyahət etdi.",
    ]
    azerbaijani_queries = [
        "Tortlar un, portağal və su ilə hazırlanır.",
        "Piroqlar mənim gəldiyim yerdə yoxa çıxdı.",
        "O, çay üzərindən tort apardı.",
    ]
    russian_corpus = [
        "Я люблю тортики больше, чем яблоки.",
        "Апельсины лучше пирогов летом.",
        "Яблочные сады цветут у дороги.",
        "Он ехал через реку на рынок.",
    ]
    russian_queries = [
        "Тортики делают из муки, апельсинов и воды.",
        "Пироги исчезли там, где я появился.",
        "Он вез тортик через реку.",
    ]

    languages = [
        ("english", english_corpus, english_queries),
        ("azerbaijani", azerbaijani_corpus, azerbaijani_queries),
        ("russian", russian_corpus, russian_queries),
    ]

    for lang, corpus, queries in languages:
        print(f"\n=== Processing {lang.upper()} language ===")
        print(f"Corpus ({len(corpus)} documents):")
        for i, doc in enumerate(corpus):
            print(f"  DOC {i}: {doc}")
        print(f"Queries to compare ({len(queries)}):")
        for i, query in enumerate(queries):
            print(f"  QUERY {i}: {query}")

        tfidf_model = TFIDF(corpus, language=lang)
        tfidf_model.preprocess_corpus()
        tfidf_model.build_terms()
        tfidf_matrix = tfidf_model.tfidf()

        print("\nTerms:", tfidf_model.terms)
        print("TF-IDF Matrix:\n", tfidf_matrix)

        print("\nCosine Similarity Results:")
        for i, query in enumerate(queries):
            print(f"\nQUERY {i}: {query}")
            similarities = tfidf_model.cosine_similarity(query)
            for j, sim in enumerate(similarities):
                print(f"Similarity with DOC {j} = {sim:.4f}")


if __name__ == "__main__":
    main()
