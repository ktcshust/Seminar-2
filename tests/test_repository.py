"""Interchangeability of the two persistence backends.

The persistence layer is the project's central non-functional requirement: the
in-memory and the file backend must be indistinguishable to a caller.  Every
behavioural test below therefore runs against *both* implementations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.repository import (
    DatasetNotFoundError,
    FileArtifactRepository,
    FileDatasetRepository,
    InMemoryArtifactRepository,
    InMemoryDatasetRepository,
    RepositoryError,
    get_artifact_repository,
    get_dataset_repository,
)


@pytest.fixture(params=["memory", "file"])
def dataset_repository(request, tmp_path):
    """Yield each dataset backend in turn."""
    if request.param == "memory":
        return InMemoryDatasetRepository()
    return FileDatasetRepository(tmp_path / "datasets")


@pytest.fixture(params=["memory", "file"])
def artifact_repository(request, tmp_path):
    """Yield each artifact backend in turn."""
    if request.param == "memory":
        return InMemoryArtifactRepository()
    return FileArtifactRepository(tmp_path / "artifacts")


def test_round_trip_preserves_content(dataset_repository, sample_frame):
    dataset_repository.save("train", sample_frame)
    restored = dataset_repository.load("train")
    pd.testing.assert_frame_equal(restored, sample_frame)


def test_missing_dataset_raises(dataset_repository):
    with pytest.raises(DatasetNotFoundError):
        dataset_repository.load("does_not_exist")


def test_exists_and_listing(dataset_repository, sample_frame):
    assert dataset_repository.list_datasets() == []
    assert not dataset_repository.exists("train")

    dataset_repository.save("train", sample_frame)
    dataset_repository.save("validation", sample_frame.head(10))

    assert dataset_repository.exists("train")
    assert "train" in dataset_repository
    assert dataset_repository.list_datasets() == ["train", "validation"]
    assert len(dataset_repository) == 2


def test_save_overwrites(dataset_repository, sample_frame):
    dataset_repository.save("train", sample_frame)
    dataset_repository.save("train", sample_frame.head(5))
    assert len(dataset_repository.load("train")) == 5


def test_delete_is_idempotent(dataset_repository, sample_frame):
    dataset_repository.save("train", sample_frame)
    dataset_repository.delete("train")
    dataset_repository.delete("train")  # must not raise
    assert not dataset_repository.exists("train")


def test_stored_frame_is_isolated_from_the_caller(dataset_repository, sample_frame):
    """Mutating the frame after saving must not change what was stored."""
    dataset_repository.save("train", sample_frame)
    sample_frame.loc[0, "comment_text"] = "MUTATED AFTER SAVE"
    assert dataset_repository.load("train").loc[0, "comment_text"] != "MUTATED AFTER SAVE"


def test_load_or_compute_caches(dataset_repository, sample_frame):
    calls = []

    def factory():
        calls.append(1)
        return sample_frame

    dataset_repository.load_or_compute("train", factory)
    dataset_repository.load_or_compute("train", factory)
    assert len(calls) == 1


def test_artifact_round_trip(artifact_repository):
    metrics = {"mean_column_auc": 0.978, "labels": ["toxic", "threat"]}
    artifact_repository.save("metrics", metrics)
    assert artifact_repository.load("metrics") == metrics
    assert artifact_repository.list_artifacts() == ["metrics"]


def test_artifact_handles_non_json_objects(artifact_repository):
    """Fitted estimators are not JSON-serialisable and must still round-trip."""
    import numpy as np

    artifact_repository.save("array", np.arange(10))
    assert artifact_repository.load("array").tolist() == list(range(10))


def test_file_repository_rejects_path_traversal(tmp_path):
    repository = FileDatasetRepository(tmp_path)
    with pytest.raises(RepositoryError):
        repository.save("../escape", pd.DataFrame({"a": [1]}))


# --------------------------------------------------------------------------- #
# The requirement itself: switching backend changes one configuration value.
# --------------------------------------------------------------------------- #


def test_factory_honours_the_configured_backend(memory_config, file_config):
    assert isinstance(get_dataset_repository(memory_config), InMemoryDatasetRepository)
    assert isinstance(get_dataset_repository(file_config), FileDatasetRepository)
    assert isinstance(get_artifact_repository(memory_config), InMemoryArtifactRepository)
    assert isinstance(get_artifact_repository(file_config), FileArtifactRepository)


def test_business_logic_is_backend_agnostic(memory_config, file_config, sample_frame):
    """The same code path produces the same result under both backends."""

    def business_logic(repository) -> int:
        """A stand-in caller: it only knows the DatasetRepository interface."""
        repository.save("train", sample_frame)
        return int(repository.load("train")["toxic"].sum())

    assert business_logic(get_dataset_repository(memory_config)) == business_logic(
        get_dataset_repository(file_config)
    )


def test_invalid_backend_is_rejected():
    from src.config import Config

    with pytest.raises(ValueError):
        Config(persistence_backend="postgres")
