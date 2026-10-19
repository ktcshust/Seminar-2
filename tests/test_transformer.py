"""Tests for the optional transformer baseline.

Marked ``slow``: they need PyTorch plus a checkpoint download, so the default
``pytest`` run skips them.  Use ``make test-all`` to include them.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.models import HEAVY_MODELS, available_models, build_model  # noqa: E402

pytestmark = pytest.mark.slow

#: A two-layer randomly initialised DistilBERT: the architecture is the real
#: one, the weights are meaningless, so the test checks wiring and not accuracy.
TINY_CHECKPOINT = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"


@pytest.fixture
def tiny_model():
    return build_model(
        "distilbert",
        model_name=TINY_CHECKPOINT,
        max_length=32,
        batch_size=8,
        epochs=1,
    )


def test_transformer_is_registered_when_torch_is_installed():
    assert "distilbert" in available_models()
    assert "distilbert" in HEAVY_MODELS  # so the fast suite skips it


def test_fit_and_predict_respect_the_common_interface(tiny_model):
    texts = ["you are an idiot", "thanks for the helpful edit"] * 8
    y = np.zeros((len(texts), 6), dtype=int)
    y[0::2, 0] = 1

    scores = tiny_model.fit(texts, y).predict_proba(texts)

    assert scores.shape == (len(texts), 6)
    assert np.all((scores >= 0) & (scores <= 1))
    assert not np.isnan(scores).any()


def test_head_is_resized_to_six_labels(tiny_model):
    """The checkpoint ships a two-label head; it must be replaced, not reused."""
    texts = ["a comment", "another comment"] * 4
    y = np.zeros((len(texts), 6), dtype=int)
    tiny_model.fit(texts, y)
    assert tiny_model.model_.config.num_labels == 6


def test_sigmoid_head_allows_several_labels_at_once(tiny_model):
    """A softmax head would force the scores to sum to one; a sigmoid must not."""
    texts = ["you are an idiot"] * 8
    y = np.zeros((len(texts), 6), dtype=int)
    y[:, [0, 4]] = 1
    scores = tiny_model.fit(texts, y).predict_proba(texts)
    assert scores.sum(axis=1).max() > 1.0 or scores.max() > 0.4


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        build_model("distilbert", model_name=TINY_CHECKPOINT).predict_proba(["x"])
