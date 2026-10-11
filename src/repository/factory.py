"""Repository factory -- the single place where a backend is chosen.

Business logic asks the factory for a :class:`~src.repository.base.
DatasetRepository`; it never imports a concrete implementation.  Switching the
whole application from files to memory is therefore one configuration change:

    >>> from src.config import Config
    >>> repo = get_dataset_repository(Config(persistence_backend="memory"))
"""

from __future__ import annotations

from pathlib import Path

from src.config import CONFIG, Config
from src.repository.base import ArtifactRepository, DatasetRepository
from src.repository.file_repository import (
    FileArtifactRepository,
    FileDatasetRepository,
)
from src.repository.memory_repository import (
    InMemoryArtifactRepository,
    InMemoryDatasetRepository,
)

#: In-memory repositories are process-wide singletons, otherwise two callers
#: asking the factory for "the" repository would silently get separate stores.
_MEMORY_DATASETS = InMemoryDatasetRepository()
_MEMORY_ARTIFACTS = InMemoryArtifactRepository()


def get_dataset_repository(
    config: Config | None = None, root: Path | str | None = None
) -> DatasetRepository:
    """Return the dataset repository selected by ``config``."""
    config = CONFIG if config is None else config
    if config.persistence_backend == "memory":
        return _MEMORY_DATASETS
    return FileDatasetRepository(
        root=root or config.processed_dir, file_format=config.file_format
    )


def get_artifact_repository(
    config: Config | None = None, root: Path | str | None = None
) -> ArtifactRepository:
    """Return the artifact repository selected by ``config``."""
    config = CONFIG if config is None else config
    if config.persistence_backend == "memory":
        return _MEMORY_ARTIFACTS
    return FileArtifactRepository(root=root or config.results_dir)


def reset_memory_repositories() -> None:
    """Empty the in-memory singletons (used by the test-suite fixtures)."""
    _MEMORY_DATASETS.clear()
    _MEMORY_ARTIFACTS.clear()
