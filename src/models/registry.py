"""Name -> factory mapping used by the experiment runner and the notebooks."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from src.models.base import BaseTextClassifier
from src.models.dummy import KeywordClassifier, PriorClassifier
from src.models.linear import (
    linear_svm,
    logistic_regression,
    logistic_regression_tuned,
    nbsvm,
    nbsvm_tuned,
)

#: Every model the CLI can run, cheapest first.
MODEL_REGISTRY: dict[str, Callable[..., BaseTextClassifier]] = {
    "prior": PriorClassifier,
    "keyword": KeywordClassifier,
    "logreg": logistic_regression,
    "nbsvm": nbsvm,
    "linear_svm": linear_svm,
    "logreg_tuned": logistic_regression_tuned,
    "nbsvm_tuned": nbsvm_tuned,
}

#: Models that serve as reference points rather than as contributions.
BASELINE_MODELS = ("prior", "keyword")

#: Hyper-parameter tuned variants searched inside each training fold.
TUNED_MODELS = ("logreg_tuned", "nbsvm_tuned")

#: Models whose cost is measured in minutes rather than seconds.  Callers that
#: sweep over "every model" -- the test suite in particular -- should iterate
#: over :func:`lightweight_models` instead, or a `pytest` run would fine-tune a
#: transformer.
HEAVY_MODELS = ("distilbert",)


def _transformer_factory(**kwargs) -> BaseTextClassifier:
    """Import the transformer lazily: loading PyTorch costs several seconds."""
    from src.models.transformer import TransformerClassifier

    return TransformerClassifier(**kwargs)


def _torch_is_available() -> bool:
    """Whether the optional deep-learning dependencies are installed."""
    return all(
        importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers")
    )


if _torch_is_available():
    MODEL_REGISTRY["distilbert"] = _transformer_factory


def build_model(name: str, **kwargs) -> BaseTextClassifier:
    """Instantiate the model registered under ``name``.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        ) from exc
    return factory(**kwargs)


def available_models() -> list[str]:
    """Return every registered model name."""
    return list(MODEL_REGISTRY)


def lightweight_models() -> list[str]:
    """Return the core registered models that train in seconds on CPU."""
    excluded = {*HEAVY_MODELS, *TUNED_MODELS}
    return [name for name in MODEL_REGISTRY if name not in excluded]
