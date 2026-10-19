"""Metrics, threshold tuning, cross validation and result analysis."""

from src.evaluation.bias import (
    DEFAULT_IDENTITY_TERMS,
    error_examples,
    identity_term_report,
)
from src.evaluation.cross_validation import (
    CrossValidationResult,
    cross_validate_model,
    make_multilabel_folds,
    multilabel_train_test_split,
)
from src.evaluation.metrics import (
    evaluate,
    hierarchy_violations,
    overall_metrics,
    per_label_metrics,
)
from src.evaluation.statistics import (
    PairedComparison,
    comparison_matrix,
    paired_comparison,
)
from src.evaluation.thresholds import describe_thresholds, optimise_thresholds

__all__ = [
    "CrossValidationResult",
    "DEFAULT_IDENTITY_TERMS",
    "PairedComparison",
    "comparison_matrix",
    "cross_validate_model",
    "describe_thresholds",
    "error_examples",
    "evaluate",
    "hierarchy_violations",
    "identity_term_report",
    "make_multilabel_folds",
    "multilabel_train_test_split",
    "optimise_thresholds",
    "overall_metrics",
    "paired_comparison",
    "per_label_metrics",
]
