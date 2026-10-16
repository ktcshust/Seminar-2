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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.models.base import BaseTextClassifier
from src.preprocessing.cleaner import TextCleaner, cleaning_ablation_cleaners
from src.preprocessing.vectorizers import (
    build_char_vectorizer,
    build_union_vectorizer,
    build_word_vectorizer,
)

#: Candidate regularisation strengths searched inside each training fold.
DEFAULT_C_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


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


def _estimator_scores(estimator: object, design: sp.csr_matrix) -> np.ndarray:
    """Return continuous [0, 1] scores from a fitted binary estimator."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(design)[:, 1]
    return _sigmoid(estimator.decision_function(design))


def _set_estimator_c(estimator: object, c_value: float) -> object:
    """Set regularisation parameter ``C`` on a linear estimator in place."""
    if hasattr(estimator, "set_params"):
        estimator.set_params(C=c_value)
    elif hasattr(estimator, "C"):
        estimator.C = c_value
    return estimator


def _select_best_c(
    estimator_factory: Callable[[], object],
    features: sp.csr_matrix,
    target: np.ndarray,
    c_grid: Sequence[float],
    use_nb_features: bool = False,
    inner_val_fraction: float = 0.15,
    random_state: int = 42,
) -> float:
    """Select the per-label regularisation ``C`` on an inner training split.

    The hold-out split is carved strictly from the training fold so that
    validation fold examples are never seen during hyper-parameter selection.
    When a rare label has too few positive examples to stratify into both
    inner subsets, the default ``C`` of ``estimator_factory()`` is returned.
    """
    candidates = [float(c) for c in c_grid]
    default_c = float(
        getattr(estimator_factory(), "C", candidates[len(candidates) // 2])
    )
    if len(candidates) == 1:
        return candidates[0]

    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    if positives < 2 or negatives < 2:
        return default_c

    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(target)),
            test_size=inner_val_fraction,
            stratify=target,
            random_state=random_state,
        )
    except ValueError:
        return default_c

    y_train, y_val = target[train_idx], target[val_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return default_c

    x_train = features[train_idx]
    x_val = features[val_idx]
    if use_nb_features:
        ratio = _nb_log_count_ratio(x_train, y_train)
        x_train = x_train.multiply(ratio).tocsr()
        x_val = x_val.multiply(ratio).tocsr()

    best_c = default_c
    best_auc = -1.0
    for candidate in candidates:
        estimator = _set_estimator_c(estimator_factory(), candidate)
        estimator.fit(x_train, y_train)
        scores = _estimator_scores(estimator, x_val)
        auc = float(roc_auc_score(y_val, scores))
        if auc > best_auc:
            best_auc = auc
            best_c = candidate
    return best_c


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
        label_names: names of the target columns.
        c_grid: optional candidate ``C`` values searched per label on an inner
            hold-out split carved from the training fold.
        inner_val_fraction: fraction of the training fold reserved for inner
            ``C`` selection when ``c_grid`` is provided.
        random_state: seed used for the inner hold-out split.
    """

    def __init__(
        self,
        name: str,
        estimator_factory: Callable[[], object],
        vectorizer_factory: Callable[[], object] = build_union_vectorizer,
        use_nb_features: bool = False,
        cleaner: TextCleaner | None = None,
        label_names: Sequence[str] | None = None,
        c_grid: Sequence[float] | None = None,
        inner_val_fraction: float = 0.15,
        random_state: int = 42,
    ) -> None:
        super().__init__(label_names=label_names)
        self.name = name
        self.estimator_factory = estimator_factory
        self.vectorizer_factory = vectorizer_factory
        self.use_nb_features = use_nb_features
        self.cleaner = TextCleaner() if cleaner is None else cleaner
        self.c_grid = tuple(float(c) for c in c_grid) if c_grid is not None else None
        self.inner_val_fraction = inner_val_fraction
        self.random_state = random_state
        self.vectorizer_ = None
        self.estimators_: list[object] = []
        self.nb_ratios_: list[np.ndarray] = []
        self.best_c_: list[float] = []

    def fit(self, texts: Sequence[str], y: np.ndarray) -> PerLabelLinearClassifier:
        """Fit the shared vectoriser, then one binary estimator per label."""
        y = np.asarray(y)
        cleaned = self.cleaner.transform(texts)
        self.vectorizer_ = self.vectorizer_factory()
        features = sp.csr_matrix(self.vectorizer_.fit_transform(cleaned))

        self.estimators_ = []
        self.nb_ratios_ = []
        self.best_c_ = []
        for index in range(y.shape[1]):
            target = y[:, index]
            if self.use_nb_features:
                ratio = _nb_log_count_ratio(features, target)
                design = features.multiply(ratio).tocsr()
            else:
                ratio = None
                design = features

            estimator = self.estimator_factory()
            default_c = float(getattr(estimator, "C", 1.0))
            if len(np.unique(target)) < 2:
                # A fold can miss a rare label entirely (`threat` is 0.3% of the
                # corpus).  Degrade to the constant prior instead of crashing.
                estimator = _ConstantEstimator(float(target.mean()))
                chosen_c = default_c
            else:
                if self.c_grid:
                    chosen_c = _select_best_c(
                        self.estimator_factory,
                        features,
                        target,
                        self.c_grid,
                        use_nb_features=self.use_nb_features,
                        inner_val_fraction=self.inner_val_fraction,
                        random_state=self.random_state + index,
                    )
                    _set_estimator_c(estimator, chosen_c)
                else:
                    chosen_c = default_c
                estimator.fit(design, target)
            self.estimators_.append(estimator)
            self.nb_ratios_.append(ratio)
            self.best_c_.append(chosen_c)
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
            scores[:, index] = _estimator_scores(estimator, design)
        return scores

    def get_params(self) -> dict[str, object]:
        """Return the hyper-parameters recorded alongside the metrics."""
        params: dict[str, object] = {
            "name": self.name,
            "use_nb_features": self.use_nb_features,
            "estimator": type(self.estimator_factory()).__name__,
        }
        if self.c_grid is not None:
            params["c_grid"] = list(self.c_grid)
        if self.best_c_:
            params["best_c"] = list(self.best_c_)
        return params


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


def logistic_regression_tuned(
    c_grid: Sequence[float] = DEFAULT_C_GRID, **kwargs
) -> PerLabelLinearClassifier:
    """Logistic regression with per-label ``C`` tuned inside the training fold."""
    return PerLabelLinearClassifier(
        name="logreg_tuned",
        estimator_factory=lambda: LogisticRegression(
            C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
        ),
        c_grid=c_grid,
        **kwargs,
    )


def nbsvm(**kwargs) -> PerLabelLinearClassifier:
    """NB-SVM: logistic regression on NB-rescaled TF-IDF features."""
    return PerLabelLinearClassifier(
        name="nbsvm",
        estimator_factory=lambda: LogisticRegression(
            C=4.0, solver="liblinear", dual=True, max_iter=1000
        ),
        use_nb_features=True,
        **kwargs,
    )


def nbsvm_tuned(
    c_grid: Sequence[float] = DEFAULT_C_GRID, **kwargs
) -> PerLabelLinearClassifier:
    """NB-SVM with per-label ``C`` tuned inside the training fold."""
    return PerLabelLinearClassifier(
        name="nbsvm_tuned",
        estimator_factory=lambda: LogisticRegression(
            C=4.0, solver="liblinear", dual=True, max_iter=1000
        ),
        use_nb_features=True,
        c_grid=c_grid,
        **kwargs,
    )


def linear_svm(**kwargs) -> PerLabelLinearClassifier:
    """Linear SVM; scores are uncalibrated decision values squashed to [0, 1]."""
    return PerLabelLinearClassifier(
        name="linear_svm",
        estimator_factory=lambda: LinearSVC(C=0.5, class_weight="balanced"),
        **kwargs,
    )


def logistic_regression_word(**kwargs) -> PerLabelLinearClassifier:
    """Word-only TF-IDF + one-vs-rest logistic regression."""
    return PerLabelLinearClassifier(
        name="logreg_word",
        estimator_factory=lambda: LogisticRegression(
            C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
        ),
        vectorizer_factory=build_word_vectorizer,
        **kwargs,
    )


def logistic_regression_char(**kwargs) -> PerLabelLinearClassifier:
    """Character-only TF-IDF + one-vs-rest logistic regression."""
    return PerLabelLinearClassifier(
        name="logreg_char",
        estimator_factory=lambda: LogisticRegression(
            C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
        ),
        vectorizer_factory=build_char_vectorizer,
        **kwargs,
    )


def feature_ablation_models(**kwargs) -> dict[str, PerLabelLinearClassifier]:
    """Return word-only, char-only and union logistic regression classifiers."""
    return {
        "logreg_word": logistic_regression_word(**kwargs),
        "logreg_char": logistic_regression_char(**kwargs),
        "logreg": logistic_regression(**kwargs),
    }


def cleaning_ablation_models(**kwargs) -> dict[str, PerLabelLinearClassifier]:
    """Return one logistic regression classifier per cleaning ablation variant."""
    models: dict[str, PerLabelLinearClassifier] = {}
    for variant, cleaner in cleaning_ablation_cleaners().items():
        model = logistic_regression(cleaner=cleaner, **kwargs)
        model.name = f"logreg_{variant}"
        models[model.name] = model
    return models

