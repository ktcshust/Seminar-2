"""Tests for dataset loading, validation and subsampling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import LABEL_COLUMNS
from src.dataio import (
    SchemaError,
    generate_synthetic_dataset,
    load_training_data,
    stratified_subsample,
    validate_labelled_frame,
)
from src.repository import InMemoryDatasetRepository


def test_synthetic_dataset_has_the_jigsaw_schema(sample_frame):
    assert {"id", "comment_text", *LABEL_COLUMNS} <= set(sample_frame.columns)
    validate_labelled_frame(sample_frame)


def test_synthetic_dataset_reproduces_the_label_hierarchy(sample_frame):
    """`severe_toxic` is a strict subset of `toxic` in the real corpus too."""
    severe = sample_frame["severe_toxic"] == 1
    assert (sample_frame.loc[severe, "toxic"] == 1).all()


def test_synthetic_dataset_is_imbalanced(sample_frame):
    assert 0.02 < sample_frame["toxic"].mean() < 0.25
    assert sample_frame["threat"].mean() < sample_frame["insult"].mean()


def test_synthetic_dataset_is_reproducible():
    first = generate_synthetic_dataset(100, random_state=3)
    second = generate_synthetic_dataset(100, random_state=3)
    pd.testing.assert_frame_equal(first, second)


def test_validation_rejects_a_missing_label():
    frame = generate_synthetic_dataset(20).drop(columns=["threat"])
    with pytest.raises(SchemaError, match="threat"):
        validate_labelled_frame(frame)


def test_validation_rejects_unscored_rows():
    """Kaggle marks unscored test rows with -1; they must never reach a metric."""
    frame = generate_synthetic_dataset(20)
    frame.loc[0, "toxic"] = -1
    with pytest.raises(SchemaError, match="not binary"):
        validate_labelled_frame(frame)


def test_loader_falls_back_to_synthetic_when_allowed(memory_config):
    repository = InMemoryDatasetRepository()
    frame = load_training_data(
        config=memory_config,
        repository=repository,
        allow_synthetic=True,
        n_synthetic=50,
    )
    assert len(frame) == 50
    assert repository.exists("train_synthetic")


def test_loader_refuses_to_invent_data_by_default(memory_config):
    with pytest.raises(FileNotFoundError, match="kaggle auth login"):
        load_training_data(
            config=memory_config,
            repository=InMemoryDatasetRepository(),
            allow_synthetic=False,
        )


def test_loader_uses_the_cache_on_the_second_call(memory_config):
    repository = InMemoryDatasetRepository()
    kwargs = {
        "config": memory_config,
        "repository": repository,
        "allow_synthetic": True,
    }
    first = load_training_data(n_synthetic=40, **kwargs)
    second = load_training_data(n_synthetic=999, **kwargs)
    assert len(second) == len(first) == 40  # the second size was ignored


def test_subsample_preserves_the_flagged_ratio(sample_frame):
    original = (sample_frame[list(LABEL_COLUMNS)].sum(axis=1) > 0).mean()
    subsample = stratified_subsample(sample_frame, 200)
    reduced = (subsample[list(LABEL_COLUMNS)].sum(axis=1) > 0).mean()
    assert len(subsample) == 200
    assert abs(original - reduced) < 0.05


def test_subsample_larger_than_the_frame_returns_everything(sample_frame):
    assert len(stratified_subsample(sample_frame, 10_000)) == len(sample_frame)


def test_synthetic_data_is_cached_under_a_separate_name(memory_config):
    """A smoke run must not leave a fake `train` for a later real run to reuse."""
    from src.dataio import SYNTHETIC_TRAIN_DATASET, TRAIN_DATASET

    repository = InMemoryDatasetRepository()
    load_training_data(
        config=memory_config,
        repository=repository,
        allow_synthetic=True,
        n_synthetic=40,
    )
    assert repository.exists(SYNTHETIC_TRAIN_DATASET)
    assert not repository.exists(TRAIN_DATASET)
