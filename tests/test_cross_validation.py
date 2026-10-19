"""Tests for fold construction and the cross-validation driver."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import LABEL_COLUMNS
from src.dataio import label_matrix, texts
from src.evaluation import cross_validate_model, make_multilabel_folds
from src.evaluation.statistics import paired_comparison
from src.models import build_model


def test_folds_partition_the_dataset(sample_frame):
    y = label_matrix(sample_frame)
    folds, strategy = make_multilabel_folds(y, n_splits=5)

    assert strategy in {"iterative", "label_powerset"}
    assert len(folds) == 5

    validation_union = np.concatenate([validation for _, validation in folds])
    assert sorted(validation_union) == list(range(len(y)))
    assert len(set(validation_union)) == len(y)  # every row validated exactly once

    for train, validation in folds:
        assert not set(train) & set(validation)


def test_folds_keep_rare_labels_present(sample_frame):
    """The reason for stratifying: every fold should see some positive example."""
    y = label_matrix(sample_frame)
    folds, _ = make_multilabel_folds(y, n_splits=5)

    frequent = y.sum(axis=0) > 5 * 5  # at least one per fold is arithmetically possible
    for _, validation in folds:
        present = y[validation].sum(axis=0) > 0
        assert present[frequent].all()


def test_folds_are_deterministic(sample_frame):
    y = label_matrix(sample_frame)
    first, _ = make_multilabel_folds(y, n_splits=3, random_state=7)
    second, _ = make_multilabel_folds(y, n_splits=3, random_state=7)
    for (train_a, _), (train_b, _) in zip(first, second, strict=True):
        assert np.array_equal(train_a, train_b)


def test_cross_validation_produces_one_metric_set_per_fold(sample_frame):
    result = cross_validate_model(
        model_factory=lambda: build_model("logreg"),
        texts=texts(sample_frame).tolist(),
        y=label_matrix(sample_frame),
        label_names=LABEL_COLUMNS,
        n_splits=3,
    )
    assert len(result.fold_metrics) == 3
    assert len(result.fit_seconds) == 3
    assert result.oof_scores.shape == (len(sample_frame), 6)
    assert not np.isnan(result.oof_scores).any()  # every row got a prediction
    assert set(result.summary.index) >= {"mean_column_auc_mean", "mean_column_auc_std"}
    assert list(result.label_summary.index) == list(LABEL_COLUMNS)


def test_cross_validation_records_both_threshold_views(sample_frame):
    """The tuned and the fixed-0.5 view come from one pass over the folds."""
    result = cross_validate_model(
        model_factory=lambda: build_model("logreg"),
        texts=texts(sample_frame).tolist(),
        y=label_matrix(sample_frame),
        label_names=LABEL_COLUMNS,
        n_splits=3,
    )
    assert len(result.fold_metrics_at_half) == 3
    assert len(result.fold_label_metrics_at_half) == 3
    assert list(result.label_summary_at_half.index) == list(LABEL_COLUMNS)

    # The 0.5 view is scored at exactly 0.5, the tuned view generally is not.
    assert (result.label_summary_at_half["threshold"] == 0.5).all()

    # Both views rank the same predictions, so AUC -- which ignores the
    # threshold -- must be identical; only the F1 columns may differ.
    assert result.summary_at_half["mean_column_auc_mean"] == pytest.approx(
        result.summary["mean_column_auc_mean"]
    )

    stored = result.to_dict()
    assert "summary_at_half" in stored
    assert "label_summary_at_half" in stored


def test_out_of_fold_predictions_cover_every_row(sample_frame):
    """Each row is binarised with the thresholds of the fold that predicted it."""
    result = cross_validate_model(
        model_factory=lambda: build_model("logreg"),
        texts=texts(sample_frame).tolist(),
        y=label_matrix(sample_frame),
        label_names=LABEL_COLUMNS,
        n_splits=3,
    )
    predictions = result.oof_predictions
    assert predictions.shape == (len(sample_frame), 6)
    # -1 is the "never filled in" sentinel; no row may keep it.
    assert (predictions >= 0).all()
    assert set(np.unique(predictions)) <= {0, 1}


def test_a_real_model_beats_the_prior_baseline(sample_frame):
    """A sanity check: if this fails, the pipeline is wired wrong."""
    shared_folds, strategy = make_multilabel_folds(label_matrix(sample_frame), n_splits=3)
    common = {
        "texts": texts(sample_frame).tolist(),
        "y": label_matrix(sample_frame),
        "label_names": LABEL_COLUMNS,
        "folds": shared_folds,
        "strategy": strategy,
    }
    model = cross_validate_model(lambda: build_model("logreg"), **common)
    baseline = cross_validate_model(lambda: build_model("prior"), **common)

    assert (
        model.summary["mean_column_auc_mean"]
        > baseline.summary["mean_column_auc_mean"]
    )
    # The prior carries no information, so its AUC must sit at chance level.
    assert baseline.summary["mean_column_auc_mean"] == np.float64(0.5)


def test_paired_comparison_reports_direction_and_p_value():
    better = np.array([0.98, 0.97, 0.98, 0.99, 0.98])
    worse = np.array([0.95, 0.94, 0.96, 0.95, 0.94])
    comparison = paired_comparison(better, worse, "a", "b")
    assert comparison.mean_difference > 0
    assert comparison.n_folds == 5
    assert 0.0 <= comparison.wilcoxon_p <= 1.0


def test_paired_comparison_on_identical_scores_is_not_significant():
    scores = np.array([0.9, 0.91, 0.89])
    comparison = paired_comparison(scores, scores, "a", "a")
    assert comparison.wilcoxon_p == 1.0
    assert not comparison.significant
