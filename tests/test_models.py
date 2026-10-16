"""Tests for the classifiers and the model registry."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import LABEL_COLUMNS
from src.dataio import label_matrix, texts
from src.models import (
    ABLATION_MODELS,
    TUNED_MODELS,
    build_model,
    cleaning_ablation_models,
    feature_ablation_models,
    lightweight_models,
)


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


@pytest.mark.parametrize("name", TUNED_MODELS)
def test_per_label_c_tuning_selects_from_grid_inside_fold(name, sample_frame):
    grid = (0.25, 1.0, 4.0)
    comment_text = texts(sample_frame).tolist()
    y = label_matrix(sample_frame)

    model = build_model(name, c_grid=grid, label_names=LABEL_COLUMNS)
    model.fit(comment_text, y)
    scores = model.predict_proba(comment_text)

    assert scores.shape == (len(comment_text), len(LABEL_COLUMNS))
    assert len(model.best_c_) == len(LABEL_COLUMNS)
    assert set(model.best_c_) <= set(grid)
    params = model.get_params()
    assert params["c_grid"] == list(grid)
    assert params["best_c"] == model.best_c_


def test_feature_ablation_models_cover_word_char_and_union(sample_frame):
    models = feature_ablation_models(label_names=LABEL_COLUMNS)
    assert set(models) == {*ABLATION_MODELS, "logreg"}

    comment_text = texts(sample_frame).tolist()
    y = label_matrix(sample_frame)
    for name in ABLATION_MODELS:
        fitted = build_model(name, label_names=LABEL_COLUMNS).fit(comment_text, y)
        scores = fitted.predict_proba(comment_text)
        assert scores.shape == (len(comment_text), len(LABEL_COLUMNS))
        assert np.all((scores >= 0) & (scores <= 1))


def test_cleaning_ablation_models_fit_and_predict():
    models = cleaning_ablation_models(label_names=LABEL_COLUMNS)
    assert "logreg_full" in models
    assert "logreg_none" in models
    assert "logreg_no_normalise_leet" in models

    toxic = ["you are an id!ot m0ron"] * 20
    clean = ["thanks for the helpful edit"] * 20
    y = np.zeros((40, 6), dtype=int)
    y[:20, 0] = 1

    for name in ("logreg_full", "logreg_none"):
        scores = models[name].fit(toxic + clean, y).predict_proba(
            ["you are an id!ot", "thanks for the edit"]
        )
        assert scores.shape == (2, 6)
        assert scores[0, 0] > scores[1, 0]


def test_unknown_model_name_is_reported_clearly():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("transformer_xl")

