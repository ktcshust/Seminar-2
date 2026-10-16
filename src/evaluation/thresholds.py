"""Per-label decision thresholds.

A single 0.5 cut-off is inherited from balanced binary problems and is a poor
fit here: with a 0.3% positive rate, the F1 of ``threat`` roughly doubles once
its threshold is tuned.  Thresholds are always chosen on training-fold
predictions and applied to the held-out fold, never fitted on the data they are
scored on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score


def optimise_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_candidates: int = 101,
    minimum: float = 0.01,
    maximum: float = 0.99,
) -> np.ndarray:
    """Return the per-label threshold maximising F1 on the given predictions.

    Args:
        y_true: binary matrix ``(n_samples, n_labels)``.
        y_score: predicted scores, same shape.
        n_candidates: size of the grid searched between ``minimum`` and
            ``maximum``.
        minimum: smallest candidate threshold.
        maximum: largest candidate threshold.

    Returns:
        An array of ``n_labels`` thresholds.  Labels with no positive example
        keep the neutral 0.5.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    grid = np.linspace(minimum, maximum, n_candidates)

    thresholds = np.full(y_true.shape[1], 0.5)
    for index in range(y_true.shape[1]):
        column_true = y_true[:, index]
        if column_true.sum() == 0:
            continue
        column_score = y_score[:, index]
        scores = [
            f1_score(column_true, (column_score >= cut).astype(int), zero_division=0)
            for cut in grid
        ]
        thresholds[index] = float(grid[int(np.argmax(scores))])
    return thresholds


def describe_thresholds(
    thresholds: np.ndarray, label_names: Sequence[str]
) -> dict[str, float]:
    """Return the thresholds as a label-keyed dictionary, ready for JSON."""
    return {
        label: float(value) for label, value in zip(label_names, thresholds, strict=True)
    }
