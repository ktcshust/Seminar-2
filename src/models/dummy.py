"""Trivial baselines.

Reporting a mean AUC of 0.97 means little until it is contrasted with what a
model that has learnt nothing achieves.  On a corpus where 90% of the comments
are clean, always predicting "not toxic" already reaches 90% accuracy -- which
is precisely why the report uses AUC and per-label F1 instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.models.base import BaseTextClassifier


class PriorClassifier(BaseTextClassifier):
    """Predict the training frequency of each label, ignoring the text.

    Its AUC is 0.5 by construction and its accuracy is high, which makes it the
    clearest possible illustration of why accuracy is the wrong metric here.
    """

    name = "prior"

    def fit(self, texts: Sequence[str], y: np.ndarray) -> PriorClassifier:
        """Record the training frequency of each label."""
        self.priors_ = np.asarray(y).mean(axis=0)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return the recorded priors, identical for every comment."""
        if not hasattr(self, "priors_"):
            raise RuntimeError("Call fit() before predict_proba().")
        return np.tile(self.priors_, (len(list(texts)), 1))


class KeywordClassifier(BaseTextClassifier):
    """Flag a comment when it contains a word from a fixed profanity list.

    A rule-based system of this kind is what a team would ship without machine
    learning, so it is the honest reference point for "is the model worth it".
    """

    name = "keyword"

    #: Kept short on purpose; the point is the method, not the word list.
    DEFAULT_TERMS = (
        "idiot", "moron", "stupid", "fool", "loser", "shit", "fuck",
        "bitch", "bastard", "damn", "hell", "kill", "die",
    )

    def __init__(self, terms: Sequence[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.terms = tuple(terms) if terms else self.DEFAULT_TERMS

    def fit(self, texts: Sequence[str], y: np.ndarray) -> KeywordClassifier:
        """Learn how often a keyword match implies each label."""
        # The rule is fixed; only the per-label hit rate is learnt, so that the
        # score reflects how often a match really implies each label.
        y = np.asarray(y)
        matched = self._matches(texts)
        self.hit_rates_ = np.array(
            [
                y[matched, index].mean() if matched.any() else 0.0
                for index in range(y.shape[1])
            ]
        )
        self.miss_rates_ = np.array(
            [
                y[~matched, index].mean() if (~matched).any() else 0.0
                for index in range(y.shape[1])
            ]
        )
        return self

    def _matches(self, texts: Sequence[str]) -> np.ndarray:
        lowered = [str(text).lower() for text in texts]
        return np.array(
            [any(term in text for term in self.terms) for text in lowered], dtype=bool
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return the hit rate for matching comments, the miss rate otherwise."""
        if not hasattr(self, "hit_rates_"):
            raise RuntimeError("Call fit() before predict_proba().")
        matched = self._matches(texts)
        return np.where(matched[:, None], self.hit_rates_, self.miss_rates_)
