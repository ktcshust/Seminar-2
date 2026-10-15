"""Tests for the classifiers and the model registry."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import LABEL_COLUMNS
from src.dataio import label_matrix, texts
from src.models import build_model, lightweight_models


# Only the fast models: the transformer has its own (slow-marked) suite.
@pytest.mark.parametrize("name", lightweight_models())
def test_every_model_honours_the_interface(name, sample_frame):
    comment_text = texts(sample_frame).tolist()
    y = label_matrix(sample_frame)

    model = build_model(name, label_names=LABEL_COLUMNS)
    model.fit(comment_text, y)
    scores = model.predict_proba(comment_text)

    assert scores.shape == (len(comment_text), len(LABEL_COLUMNS))
    assert np.all((scores >= 0) & (scores <= 1))
    assert not np.isnan(scores).any()

    predictions = model.predict(comment_text)
    assert set(np.unique(predictions)) <= {0, 1}


def test_predict_accepts_per_label_thresholds(sample_frame):
    model = build_model("logreg").fit(
        texts(sample_frame).tolist(), label_matrix(sample_frame)
    )
    everything = model.predict(texts(sample_frame).tolist(), thresholds=np.zeros(6))
    nothing = model.predict(texts(sample_frame).tolist(), thresholds=np.ones(6) + 1e-9)
    assert everything.sum() > nothing.sum()
    assert nothing.sum() == 0


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        build_model("logreg").predict_proba(["anything"])


def test_model_learns_something_on_separable_data():
    toxic = ["you are an idiot moron"] * 40
    clean = ["thanks for the helpful edit"] * 40
    y = np.zeros((80, 6), dtype=int)
    y[:40, 0] = 1

    model = build_model("logreg").fit(toxic + clean, y)
    scores = model.predict_proba(["you are an idiot", "thanks for the edit"])
    assert scores[0, 0] > scores[1, 0]


def test_single_class_label_does_not_crash_training():
    """A rare label can be entirely absent from a training fold."""
    documents = ["some comment", "another comment", "a third one"] * 5
    y = np.zeros((15, 6), dtype=int)  # no positive anywhere
    model = build_model("nbsvm").fit(documents, y)
    assert model.predict_proba(documents).shape == (15, 6)


def test_unknown_model_name_is_reported_clearly():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("transformer_xl")
