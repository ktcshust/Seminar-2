"""Linear binary-relevance models over TF-IDF features.

One class covers the three classic baselines of the challenge, because they
differ only in the binary estimator and in whether the features are rescaled by
the Naive Bayes log-count ratio:

===============  ====================  ==================
model            estimator             NB rescaling
===============  ====================  ==================
``logreg``       LogisticRegression    no
``linear_svm``   LinearSVC             no
``nbsvm``        LogisticRegression    yes
===============  ====================  ==================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseTextClassifier
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.vectorizers import build_union_vectorizer


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic squashing of a decision function."""
    return np.where(
        values >= 0,
        1.0 / (1.0 + np.exp(-np.clip(values, -30, 30))),
        np.exp(np.clip(values, -30, 30)) / (1.0 + np.exp(np.clip(values, -30, 30))),
    )


def _nb_log_count_ratio(matrix: sp.csr_matrix, y: np.ndarray, alpha: float = 1.0):
    """Return the Naive Bayes log-count ratio ``r`` of Wang & Manning (2012).

    ``r_i = log( P(feature_i | positive) / P(feature_i | negative) )`` with
    Laplace smoothing.  Multiplying the TF-IDF matrix by ``r`` before fitting a
    linear model is the "NB-SVM" trick that is remarkably strong on this task.
    """
    positive = np.asarray(matrix[y == 1].sum(axis=0)).ravel() + alpha
    negative = np.asarray(matrix[y == 0].sum(axis=0)).ravel() + alpha
    return np.log((positive / positive.sum()) / (negative / negative.sum()))


class PerLabelLinearClassifier(BaseTextClassifier):
    """Fit one linear binary classifier per label on shared TF-IDF features.

    The vectoriser is fitted once on the training fold and reused for all six
    labels; only the estimators are per-label.  ``class_weight="balanced"`` is
    the default because five of the six labels have a positive rate below 6%.

    Args:
        name: identifier of the configuration.
        estimator_factory: callable returning an unfitted binary estimator.
        vectorizer_factory: callable returning an unfitted vectoriser.
        use_nb_features: rescale the features by the NB log-count ratio.
        cleaner: text normaliser; a default :class:`TextCleaner` if omitted.
    """

    def __init__(
        self,
        name: str,
        estimator_factory: Callable[[], object],
        vectorizer_factory: Callable[[], object] = build_union_vectorizer,
        use_nb_features: bool = False,
        cleaner: TextCleaner | None = None,
        label_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(label_names=label_names)
        self.name = name
        self.estimator_factory = estimator_factory
        self.vectorizer_factory = vectorizer_factory
        self.use_nb_features = use_nb_features
        self.cleaner = TextCleaner() if cleaner is None else cleaner
        self.vectorizer_ = None
        self.estimators_: list[object] = []
        self.nb_ratios_: list[np.ndarray] = []

    def fit(self, texts: Sequence[str], y: np.ndarray) -> PerLabelLinearClassifier:
        """Fit the shared vectoriser, then one binary estimator per label."""
        y = np.asarray(y)
        cleaned = self.cleaner.transform(texts)
        self.vectorizer_ = self.vectorizer_factory()
        features = sp.csr_matrix(self.vectorizer_.fit_transform(cleaned))

        self.estimators_ = []
        self.nb_ratios_ = []
        for index in range(y.shape[1]):
            target = y[:, index]
            if self.use_nb_features:
                ratio = _nb_log_count_ratio(features, target)
                design = features.multiply(ratio).tocsr()
            else:
                ratio = None
                design = features

            estimator = self.estimator_factory()
            if len(np.unique(target)) < 2:
                # A fold can miss a rare label entirely (`threat` is 0.3% of the
                # corpus).  Degrade to the constant prior instead of crashing.
                estimator = _ConstantEstimator(float(target.mean()))
            else:
                estimator.fit(design, target)
            self.estimators_.append(estimator)
            self.nb_ratios_.append(ratio)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return the per-label score of every comment."""
        if self.vectorizer_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        cleaned = self.cleaner.transform(texts)
        features = sp.csr_matrix(self.vectorizer_.transform(cleaned))

        scores = np.zeros((features.shape[0], len(self.estimators_)))
        for index, estimator in enumerate(self.estimators_):
            ratio = self.nb_ratios_[index]
            design = features.multiply(ratio).tocsr() if ratio is not None else features
            if hasattr(estimator, "predict_proba"):
                scores[:, index] = estimator.predict_proba(design)[:, 1]
            else:
                scores[:, index] = _sigmoid(estimator.decision_function(design))
        return scores

    def get_params(self) -> dict[str, object]:
        """Return the hyper-parameters recorded alongside the metrics."""
        return {
            "name": self.name,
            "use_nb_features": self.use_nb_features,
            "estimator": type(self.estimator_factory()).__name__,
        }


class _ConstantEstimator:
    """Fallback used when a training fold contains a single class."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, X):  # noqa: N803 - sklearn signature
        """Return the constant class prior for every row."""
        column = np.full(X.shape[0], self.probability)
        return np.column_stack([1.0 - column, column])


# --------------------------------------------------------------------------- #
# Ready-made configurations
# --------------------------------------------------------------------------- #


def logistic_regression(**kwargs) -> PerLabelLinearClassifier:
    """TF-IDF + one-vs-rest logistic regression -- the reference baseline."""
    return PerLabelLinearClassifier(
        name="logreg",
        estimator_factory=lambda: LogisticRegression(
            C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
        ),
        **kwargs,
    )
