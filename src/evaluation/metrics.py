"""Evaluation metrics for an imbalanced multi-label problem.

Accuracy is meaningless here: predicting "clean" for every comment scores about
0.90.  The report therefore uses

* **mean column-wise ROC-AUC** -- the official Kaggle metric, so the numbers are
  directly comparable with the public leaderboard;
* **average precision (PR-AUC)** -- far more informative than AUC on labels with
  a 0.3% positive rate such as ``threat``;
* **per-label precision / recall / F1** at a tuned threshold;
* **Hamming loss** and **subset accuracy** -- the two classic multi-label views,
  the first forgiving, the second strict.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC that returns NaN instead of raising on a single-class column."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision that returns NaN on a single-class column."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def per_label_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
    thresholds: np.ndarray | float = 0.5,
) -> pd.DataFrame:
    """Return one row of metrics per label.

    Args:
        y_true: binary matrix ``(n_samples, n_labels)``.
        y_score: predicted scores in [0, 1], same shape.
        label_names: column names, in the order of ``y_true``.
        thresholds: scalar or one cut-off per label.

    Returns:
        A frame indexed by label with support, positive rate, ROC-AUC, PR-AUC,
        precision, recall and F1.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    y_pred = (y_score >= thresholds).astype(int)

    rows = []
    with warnings.catch_warnings():
        # A label with no predicted positive yields an undefined precision; the
        # zero_division argument already handles it, silence the noise.
        warnings.simplefilter("ignore")
        for index, label in enumerate(label_names):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, index],
                y_pred[:, index],
                average="binary",
                zero_division=0,
            )
            rows.append(
                {
                    "label": label,
                    "support": int(y_true[:, index].sum()),
                    "positive_rate": float(y_true[:, index].mean()),
                    "roc_auc": _safe_roc_auc(y_true[:, index], y_score[:, index]),
                    "pr_auc": _safe_average_precision(
                        y_true[:, index], y_score[:, index]
                    ),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "threshold": float(thresholds[index]),
                }
            )
    return pd.DataFrame(rows).set_index("label")


def overall_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, float]:
    """Return the aggregated metrics used to rank models in the report."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    y_pred = (y_score >= thresholds).astype(int)

    label_frame = per_label_metrics(y_true, y_score, label_names, thresholds)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            # The official competition metric.
            "mean_column_auc": float(np.nanmean(label_frame["roc_auc"])),
            "mean_column_pr_auc": float(np.nanmean(label_frame["pr_auc"])),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            # Strictest possible view: every one of the six labels must match.
            "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        }


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
    thresholds: np.ndarray | float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Convenience wrapper returning ``(overall, per_label)`` in one call."""
    return (
        overall_metrics(y_true, y_score, label_names, thresholds),
        per_label_metrics(y_true, y_score, label_names, thresholds),
    )


def hierarchy_violations(
    y_pred: np.ndarray,
    label_names: Sequence[str],
    parent: str = "toxic",
    children: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Count predictions that break the annotation scheme's label hierarchy.

    In the training data every ``severe_toxic`` comment is also ``toxic``
    (the conditional probability is exactly 1), and the other minority
    labels imply ``toxic`` with probability above 0.92.  Binary relevance fits
    each label independently and is therefore free to predict a child without
    its parent, which no annotator ever did.  Counting how often that happens
    quantifies the cost of the decomposition instead of merely asserting it.

    Args:
        y_pred: binary prediction matrix ``(n_samples, n_labels)``.
        label_names: column names, in the order of ``y_pred``.
        parent: the label every child is expected to imply.
        children: labels to check; all non-parent labels when omitted.

    Returns:
        A frame indexed by child label with the number of predicted positives,
        how many of them lack the parent, and that count as a rate.
    """
    y_pred = np.asarray(y_pred)
    label_names = list(label_names)
    if parent not in label_names:
        raise ValueError(f"Parent label {parent!r} is not in {label_names}.")
    if children is None:
        children = [name for name in label_names if name != parent]

    parent_positive = y_pred[:, label_names.index(parent)] == 1
    rows = []
    for child in children:
        child_positive = y_pred[:, label_names.index(child)] == 1
        violations = int(np.sum(child_positive & ~parent_positive))
        predicted = int(child_positive.sum())
        rows.append(
            {
                "label": child,
                "predicted_positive": predicted,
                "without_parent": violations,
                "violation_rate": float(violations / predicted) if predicted else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("label")
