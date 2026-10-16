"""Classifiers, all exposing the same ``fit`` / ``predict_proba`` interface."""

from src.models.base import BaseTextClassifier
from src.models.dummy import KeywordClassifier, PriorClassifier
from src.models.linear import (
    DEFAULT_C_GRID,
    PerLabelLinearClassifier,
    linear_svm,
    logistic_regression,
    logistic_regression_tuned,
    nbsvm,
    nbsvm_tuned,
)
from src.models.registry import (
    BASELINE_MODELS,
    HEAVY_MODELS,
    MODEL_REGISTRY,
    TUNED_MODELS,
    available_models,
    build_model,
    lightweight_models,
)

__all__ = [
    "BASELINE_MODELS",
    "BaseTextClassifier",
    "DEFAULT_C_GRID",
    "HEAVY_MODELS",
    "KeywordClassifier",
    "MODEL_REGISTRY",
    "PerLabelLinearClassifier",
    "PriorClassifier",
    "TUNED_MODELS",
    "available_models",
    "build_model",
    "lightweight_models",
    "linear_svm",
    "logistic_regression",
    "logistic_regression_tuned",
    "nbsvm",
    "nbsvm_tuned",
]
