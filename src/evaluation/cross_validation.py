"""K-fold cross validation for multi-label data.

Plain ``KFold`` ignores the labels and ``StratifiedKFold`` only accepts a single
target, so neither is appropriate: with ``threat`` present in 0.3% of the
comments, an unstratified 5-fold split can easily produce a validation fold with
almost no positive example, making its AUC undefined or wildly unstable.

Two strategies are provided, in order of preference:

1. **Iterative stratification** (Sechidis et al., 2011) via the optional
   ``iterative-stratification`` package -- it balances every label across folds
   simultaneously.
2. **Label-powerset stratification** -- the fallback: each distinct combination
   of the six labels becomes a class for ``StratifiedKFold``, with combinations
   too rare to be split merged into a bucket keyed by their cardinality.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import overall_metrics, per_label_metrics
from src.evaluation.thresholds import optimise_thresholds
from src.models.base import BaseTextClassifier

LOGGER = logging.getLogger(__name__)

Fold = tuple[np.ndarray, np.ndarray]


def _powerset_strata(y: np.ndarray, n_splits: int) -> np.ndarray:
    """Return a single stratification target derived from the label matrix."""
    combinations = np.array(["".join(map(str, row)) for row in y])
    labels, counts = np.unique(combinations, return_counts=True)
    too_rare = set(labels[counts < n_splits])
    if too_rare:
        cardinality = y.sum(axis=1)
        combinations = np.where(
            np.isin(combinations, list(too_rare)),
            np.char.add("rare_", cardinality.astype(str)),
            combinations,
        )
    return combinations


def make_multilabel_folds(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[list[Fold], str]:
    """Return ``n_splits`` train/validation index pairs and the strategy used.

    Args:
        y: binary label matrix ``(n_samples, n_labels)``.
        n_splits: number of folds.
        random_state: seed, so that every model sees exactly the same folds.
        shuffle: shuffle before splitting.

    Returns:
        A ``(folds, strategy)`` pair where ``strategy`` is either
        ``"iterative"`` or ``"label_powerset"`` -- record it in the report, the
        two are not interchangeable.
    """
    y = np.asarray(y)
    placeholder = np.zeros(len(y))

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        folds = [
            (train, validation) for train, validation in splitter.split(placeholder, y)
        ]
        return folds, "iterative"
    except Exception as exc:  # pragma: no cover - depends on the environment
        LOGGER.warning(
            "Iterative stratification unavailable (%s); "
            "falling back to label-powerset stratification.",
            exc,
        )

    strata = _powerset_strata(y, n_splits)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    folds = [
        (train, validation) for train, validation in splitter.split(placeholder, strata)
    ]
    return folds, "label_powerset"


def multilabel_train_test_split(
    y: np.ndarray, test_size: float = 0.15, random_state: int = 42
) -> Fold:
    """Return one stratified ``(train, test)`` index pair."""
    n_splits = max(2, round(1.0 / max(test_size, 1e-9)))
    folds, _ = make_multilabel_folds(y, n_splits=n_splits, random_state=random_state)
    return folds[0]


@dataclass
class CrossValidationResult:
    """Everything a fold-by-fold evaluation produces."""

    model_name: str
    strategy: str
    n_splits: int
    label_names: list[str]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    fold_label_metrics: list[pd.DataFrame] = field(default_factory=list)
    #: The same two, scored at a fixed 0.5 cut-off.  Recorded in the same pass
    #: so the gain from threshold tuning costs no extra training.
    fold_metrics_at_half: list[dict[str, float]] = field(default_factory=list)
    fold_label_metrics_at_half: list[pd.DataFrame] = field(default_factory=list)
    fold_thresholds: list[np.ndarray] = field(default_factory=list)
    fit_seconds: list[float] = field(default_factory=list)
    #: Out-of-fold scores: every row predicted by the fold that did not train on it.
    oof_scores: np.ndarray | None = None
    #: The same rows binarised with the thresholds of the fold that predicted
    #: them -- what the error and label-hierarchy analyses actually inspect.
    oof_predictions: np.ndarray | None = None

    @property
    def summary(self) -> pd.Series:
        """Mean and standard deviation of every aggregated metric."""
        frame = pd.DataFrame(self.fold_metrics)
        summary = {}
        for column in frame.columns:
            summary[f"{column}_mean"] = float(frame[column].mean())
            summary[f"{column}_std"] = float(frame[column].std(ddof=1))
        summary["fit_seconds_mean"] = float(np.mean(self.fit_seconds))
        return pd.Series(summary, name=self.model_name)

    @property
    def label_summary(self) -> pd.DataFrame:
        """Per-label metrics averaged over the folds."""
        stacked = pd.concat(self.fold_label_metrics)
        return stacked.groupby(level=0).mean().loc[self.label_names]

    @property
    def summary_at_half(self) -> pd.Series:
        """Aggregated metrics at the fixed 0.5 cut-off."""
        frame = pd.DataFrame(self.fold_metrics_at_half)
        summary = {f"{column}_mean": float(frame[column].mean()) for column in frame}
        return pd.Series(summary, name=self.model_name)

    @property
    def label_summary_at_half(self) -> pd.DataFrame:
        """Per-label metrics at the fixed 0.5 cut-off, averaged over folds."""
        stacked = pd.concat(self.fold_label_metrics_at_half)
        return stacked.groupby(level=0).mean().loc[self.label_names]

    def fold_scores(self, metric: str = "mean_column_auc") -> np.ndarray:
        """Return the per-fold values of ``metric`` (input to a paired test)."""
        return np.array([fold[metric] for fold in self.fold_metrics])

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view, for the artifact repository."""
        return {
            "model_name": self.model_name,
            "strategy": self.strategy,
            "n_splits": self.n_splits,
            "label_names": self.label_names,
            "fold_metrics": self.fold_metrics,
            "fit_seconds": self.fit_seconds,
            "summary": self.summary.to_dict(),
            "label_summary": self.label_summary.to_dict(),
            "summary_at_half": self.summary_at_half.to_dict(),
            "label_summary_at_half": self.label_summary_at_half.to_dict(),
            "thresholds": [threshold.tolist() for threshold in self.fold_thresholds],
        }


def cross_validate_model(
    model_factory: Callable[[], BaseTextClassifier],
    texts: Sequence[str],
    y: np.ndarray,
    label_names: Sequence[str],
    n_splits: int = 5,
    random_state: int = 42,
    tune_thresholds: bool = True,
    inner_holdout: float = 0.15,
    folds: list[Fold] | None = None,
    strategy: str | None = None,
) -> CrossValidationResult:
    """Run k-fold cross validation and return per-fold and aggregated metrics.

    Thresholds, when tuned, are chosen on an *inner* hold-out carved out of the
    training fold and never on the validation fold, so the reported F1 is not
    optimistically biased.  The model is fitted on the remaining part of the
    training fold only; pass ``tune_thresholds=False`` to train on the full fold
    and score everything at 0.5.

    Args:
        model_factory: zero-argument callable returning an unfitted model.  A
            *factory* rather than an instance, so each fold gets a fresh model.
        texts: raw comments.
        y: binary label matrix.
        label_names: label column names in the order of ``y``.
        n_splits: number of folds.
        random_state: seed for the fold assignment.
        tune_thresholds: enable per-label threshold optimisation.
        inner_holdout: fraction of the training fold reserved for tuning.
        folds: reuse a pre-computed fold assignment -- pass the same list to
            every model so the paired statistical test is valid.
        strategy: name of the strategy that produced ``folds``.

    Returns:
        A :class:`CrossValidationResult`.
    """
    texts = np.asarray(list(texts), dtype=object)
    y = np.asarray(y)

    if folds is None:
        folds, strategy = make_multilabel_folds(
            y, n_splits=n_splits, random_state=random_state
        )

    probe = model_factory()
    result = CrossValidationResult(
        model_name=probe.name,
        strategy=strategy or "unknown",
        n_splits=len(folds),
        label_names=list(label_names),
    )
    oof = np.full((len(y), y.shape[1]), np.nan)
    oof_predicted = np.full((len(y), y.shape[1]), -1, dtype=int)

    for fold_index, (train_index, validation_index) in enumerate(folds, start=1):
        model = model_factory()

        if tune_thresholds:
            inner_fit, inner_tune = multilabel_train_test_split(
                y[train_index], test_size=inner_holdout, random_state=random_state
            )
            fit_index = train_index[inner_fit]
            tune_index = train_index[inner_tune]
        else:
            fit_index, tune_index = train_index, None

        started = time.perf_counter()
        model.fit(texts[fit_index].tolist(), y[fit_index])
        elapsed = time.perf_counter() - started

        if tune_index is not None:
            thresholds = optimise_thresholds(
                y[tune_index], model.predict_proba(texts[tune_index].tolist())
            )
        else:
            thresholds = np.full(y.shape[1], 0.5)

        scores = model.predict_proba(texts[validation_index].tolist())
        oof[validation_index] = scores
        oof_predicted[validation_index] = (scores >= thresholds).astype(int)

        result.fold_metrics.append(
            overall_metrics(y[validation_index], scores, label_names, thresholds)
        )
        result.fold_label_metrics.append(
            per_label_metrics(y[validation_index], scores, label_names, thresholds)
        )
        # Scoring the same predictions at 0.5 costs nothing and is what
        # Section "Effect of threshold tuning" compares against.
        result.fold_metrics_at_half.append(
            overall_metrics(y[validation_index], scores, label_names, 0.5)
        )
        result.fold_label_metrics_at_half.append(
            per_label_metrics(y[validation_index], scores, label_names, 0.5)
        )
        result.fold_thresholds.append(thresholds)
        result.fit_seconds.append(elapsed)

        LOGGER.info(
            "[%s] fold %d/%d  AUC=%.4f  macro-F1=%.4f  (%.1fs)",
            result.model_name,
            fold_index,
            len(folds),
            result.fold_metrics[-1]["mean_column_auc"],
            result.fold_metrics[-1]["macro_f1"],
            elapsed,
        )

    result.oof_scores = oof
    result.oof_predictions = oof_predicted
    return result
