"""Tests for the metric and threshold modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import LABEL_COLUMNS
from src.evaluation import (
    describe_thresholds,
    hierarchy_violations,
    optimise_thresholds,
    overall_metrics,
    per_label_metrics,
)


@pytest.fixture
def perfect_predictions():
    y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0]])
    return y_true, y_true.astype(float)


def test_perfect_predictions_score_one(perfect_predictions):
    y_true, y_score = perfect_predictions
    metrics = overall_metrics(y_true, y_score, LABEL_COLUMNS)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert metrics["subset_accuracy"] == pytest.approx(1.0)
    assert metrics["hamming_loss"] == pytest.approx(0.0)


def test_single_class_label_yields_nan_auc_not_an_exception():
    """`threat` is absent from small folds; the metric must degrade, not crash."""
    y_true = np.zeros((10, 6), dtype=int)
    y_score = np.random.default_rng(0).random((10, 6))
    frame = per_label_metrics(y_true, y_score, LABEL_COLUMNS)
    assert frame["roc_auc"].isna().all()


def test_per_label_frame_has_one_row_per_label():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=(50, 6))
    y_score = rng.random((50, 6))
    frame = per_label_metrics(y_true, y_score, LABEL_COLUMNS)
    assert list(frame.index) == list(LABEL_COLUMNS)
    assert frame["support"].sum() == y_true.sum()


def test_accuracy_is_misleading_on_imbalanced_data():
    """The motivation for using AUC: a constant predictor looks excellent."""
    y_true = np.zeros((1000, 6), dtype=int)
    y_true[:5, 0] = 1  # 0.5% positive rate
    y_score = np.zeros((1000, 6))
    metrics = overall_metrics(y_true, y_score, LABEL_COLUMNS)
    assert metrics["hamming_loss"] < 0.001  # looks nearly perfect
    assert metrics["macro_f1"] == pytest.approx(0.0)  # but finds nothing


def test_threshold_optimisation_beats_the_default_cutoff():
    rng = np.random.default_rng(0)
    y_true = np.zeros((400, 6), dtype=int)
    y_true[:20, 0] = 1
    # Scores that separate the classes but live far below 0.5.
    y_score = rng.uniform(0.0, 0.10, size=(400, 6))
    y_score[:20, 0] += 0.15

    from sklearn.metrics import f1_score

    tuned = optimise_thresholds(y_true, y_score)
    at_default = f1_score(y_true[:, 0], (y_score[:, 0] >= 0.5).astype(int))
    at_tuned = f1_score(y_true[:, 0], (y_score[:, 0] >= tuned[0]).astype(int))
    assert at_tuned > at_default


def test_threshold_for_empty_label_stays_neutral():
    y_true = np.zeros((20, 6), dtype=int)
    y_score = np.random.default_rng(0).random((20, 6))
    assert optimise_thresholds(y_true, y_score) == pytest.approx(np.full(6, 0.5))


def test_describe_thresholds_maps_every_label():
    described = describe_thresholds(np.arange(6) / 10, LABEL_COLUMNS)
    assert set(described) == set(LABEL_COLUMNS)


def test_hierarchy_violations_counts_children_without_their_parent():
    # Rows: a clean comment, a well-formed toxic+obscene pair, and two
    # predictions that the annotation scheme forbids -- a child on its own.
    y_pred = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    violations = hierarchy_violations(y_pred, LABEL_COLUMNS)

    assert violations.loc["severe_toxic", "without_parent"] == 1
    assert violations.loc["severe_toxic", "violation_rate"] == pytest.approx(1.0)
    # obscene is predicted twice, once with toxic and once without.
    assert violations.loc["obscene", "predicted_positive"] == 2
    assert violations.loc["obscene", "without_parent"] == 1
    assert violations.loc["obscene", "violation_rate"] == pytest.approx(0.5)
    # toxic is the parent and is never reported against itself.
    assert "toxic" not in violations.index


def test_hierarchy_violations_reports_zero_rate_for_an_unpredicted_label():
    y_pred = np.zeros((5, 6), dtype=int)
    violations = hierarchy_violations(y_pred, LABEL_COLUMNS)
    assert (violations["violation_rate"] == 0.0).all()
    assert (violations["predicted_positive"] == 0).all()


def test_hierarchy_violations_rejects_an_unknown_parent():
    with pytest.raises(ValueError, match="not in"):
        hierarchy_violations(np.zeros((2, 6), dtype=int), LABEL_COLUMNS, parent="rude")
