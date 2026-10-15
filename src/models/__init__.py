"""Classifiers, all exposing the same ``fit`` / ``predict_proba`` interface."""

from src.models.base import BaseTextClassifier
from src.models.dummy import KeywordClassifier, PriorClassifier
from src.models.linear import (
    PerLabelLinearClassifier,
    linear_svm,
    logistic_regression,
    nbsvm,
)
from src.models.registry import (
    BASELINE_MODELS,
    HEAVY_MODELS,
    MODEL_REGISTRY,
    available_models,
    build_model,
    lightweight_models,
)

__all__ = [
    "BASELINE_MODELS",
    "HEAVY_MODELS",
    "BaseTextClassifier",
    "KeywordClassifier",
    "MODEL_REGISTRY",
    "PerLabelLinearClassifier",
    "PriorClassifier",
    "available_models",
    "lightweight_models",
    "build_model",
    "linear_svm",
    "logistic_regression",
    "nbsvm",
]
