"""Common interface for every classifier in the project.

Binary relevance is the decomposition used throughout: the six labels are
predicted by six independent binary models.  It is the standard baseline for
this dataset and it makes the per-label analysis in the report straightforward,
at the cost of ignoring label correlations -- a limitation discussed explicitly
rather than hidden.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BaseTextClassifier(ABC):
    """A multi-label classifier that consumes raw text.

    Attributes:
        name: short identifier used in result tables and file names.
        label_names: the labels the model was fitted on, in column order.
    """

    name: str = "base"

    def __init__(self, label_names: Sequence[str] | None = None) -> None:
        self.label_names = list(label_names) if label_names else []

    @abstractmethod
    def fit(self, texts: Sequence[str], y: np.ndarray) -> BaseTextClassifier:
        """Fit the model.

        Args:
            texts: raw comments.
            y: binary matrix of shape ``(n_samples, n_labels)``.
        """

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-label scores of shape ``(n_samples, n_labels)`` in [0, 1].

        For models without a probabilistic output the value is a monotone
        transform of the decision function: it is a valid ranking (so AUC is
        exact) but it is *not* calibrated, which matters when picking a
        threshold.
        """

    def predict(
        self, texts: Sequence[str], thresholds: np.ndarray | float = 0.5
    ) -> np.ndarray:
        """Return binary predictions of shape ``(n_samples, n_labels)``.

        Args:
            texts: raw comments.
            thresholds: a single cut-off or one per label.  Per-label values
                come from :mod:`src.evaluation.thresholds`; the default 0.5 is
                a poor choice on a corpus this imbalanced.
        """
        scores = self.predict_proba(texts)
        return (scores >= np.asarray(thresholds)).astype(int)

    def get_params(self) -> dict[str, object]:
        """Return the hyper-parameters worth recording alongside the metrics."""
        return {"name": self.name}
