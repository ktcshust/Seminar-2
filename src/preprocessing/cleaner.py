"""Text normalisation for Wikipedia talk-page comments.

The Jigsaw comments are raw wiki markup written by anonymous editors, so they
carry IP addresses, signatures, escaped newlines and a fair amount of obfuscated
profanity (``sh!t``, ``f\u2217ck``, ``stuuuupid``).  Every step below is optional and
switched on through the constructor, which keeps the ablation study in the
report cheap to run.
"""

from __future__ import annotations

import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin

_URL = re.compile(r"https?://\S+|www\.\S+")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_USER = re.compile(r"\[\[User(?:[^\]]*)\]\]", flags=re.IGNORECASE)
_ESCAPED_NEWLINE = re.compile(r"\\[nrt]|\n|\r|\t")
_NON_ALPHA = re.compile(r"[^a-z0-9'\s]")
_REPEATED_CHAR = re.compile(r"(.)\1{2,}")
_WHITESPACE = re.compile(r"\s+")

#: Symbol substitutions used to evade word filters; safe to apply everywhere.
_SYMBOL_LEET = str.maketrans({"@": "a", "$": "s", "!": "i"})

#: Digit substitutions, applied *only* inside tokens that also contain letters.
#: Rewriting digits unconditionally would turn the year "2013" into "2oie" and
#: destroy every number in the corpus, while "b1tch" still has to become "bitch".
_DIGIT_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_MIXED_TOKEN = re.compile(r"\b(?=\w*[a-z])(?=\w*\d)\w+\b")

#: Expanded so that negations survive tokenisation ("can't" -> "can not").
_CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}


class TextCleaner(BaseEstimator, TransformerMixin):
    """A scikit-learn transformer that normalises a sequence of comments.

    Being a transformer means the cleaner can sit inside the same ``Pipeline``
    as the vectoriser and the classifier, so it is fitted and applied inside
    every cross-validation fold rather than once over the whole corpus.

    Args:
        lowercase: fold the text to lower case.
        strip_urls: replace URLs and IP addresses with a placeholder token.
        strip_wiki_markup: remove ``[[User:...]]`` signatures and escaped newlines.
        expand_contractions: rewrite English contractions.
        normalise_leet: undo the common ``@ $ !`` substitutions everywhere and
            the digit-for-letter ones inside mixed tokens (``b1tch`` -> ``bitch``,
            while ``2013`` is left alone).
        collapse_repeats: shorten runs of three or more identical characters.
        strip_punctuation: drop everything outside ``[a-z0-9']``.
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_urls: bool = True,
        strip_wiki_markup: bool = True,
        expand_contractions: bool = True,
        normalise_leet: bool = True,
        collapse_repeats: bool = True,
        strip_punctuation: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self.strip_urls = strip_urls
        self.strip_wiki_markup = strip_wiki_markup
        self.expand_contractions = expand_contractions
        self.normalise_leet = normalise_leet
        self.collapse_repeats = collapse_repeats
        self.strip_punctuation = strip_punctuation

    def clean(self, text: str) -> str:
        """Normalise a single comment."""
        text = unicodedata.normalize("NFKC", str(text))

        if self.strip_wiki_markup:
            text = _USER.sub(" ", text)
            text = _ESCAPED_NEWLINE.sub(" ", text)
        if self.strip_urls:
            text = _URL.sub(" url ", text)
            text = _IP.sub(" ipaddress ", text)
        if self.lowercase:
            text = text.lower()
        if self.expand_contractions:
            for source, target in _CONTRACTIONS.items():
                text = text.replace(source, target)
        if self.normalise_leet:
            text = text.translate(_SYMBOL_LEET)
            text = _MIXED_TOKEN.sub(
                lambda match: match.group(0).translate(_DIGIT_LEET), text
            )
        if self.strip_punctuation:
            text = _NON_ALPHA.sub(" ", text)
        if self.collapse_repeats:
            # "stuuuupid" -> "stuupid": two characters keep the emphasis signal
            # that a single one would erase.
            text = _REPEATED_CHAR.sub(r"\1\1", text)

        return _WHITESPACE.sub(" ", text).strip()

    # -- scikit-learn API -------------------------------------------------- #

    def fit(self, X, y=None) -> TextCleaner:  # noqa: N803 - sklearn signature
        """Stateless: present only to satisfy the transformer interface."""
        return self

    def transform(self, X):  # noqa: N803 - sklearn signature
        """Return the cleaned version of every element of ``X``."""
        return [self.clean(document) for document in X]
