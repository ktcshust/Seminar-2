"""Feature extraction.

Word n-grams capture the vocabulary of an insult; character n-grams capture the
spelling tricks used to hide it (``f u c k``, ``fuuuck``) and are what lifted the
classic Jigsaw baselines above 0.97 AUC.  The union of the two is the default.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_word_vectorizer(
    max_features: int = 50_000, min_df: int = 2, ngram_range: tuple[int, int] = (1, 2)
) -> TfidfVectorizer:
    """Return a sublinear TF-IDF vectoriser over word n-grams."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"\w{1,}",
    )


def build_char_vectorizer(
    max_features: int = 50_000, min_df: int = 2, ngram_range: tuple[int, int] = (2, 5)
) -> TfidfVectorizer:
    """Return a sublinear TF-IDF vectoriser over character n-grams."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def build_union_vectorizer(
    word_features: int = 50_000, char_features: int = 50_000
) -> FeatureUnion:
    """Return the horizontal concatenation of the word and character spaces."""
    return FeatureUnion(
        [
            ("word", build_word_vectorizer(max_features=word_features)),
            ("char", build_char_vectorizer(max_features=char_features)),
        ]
    )
