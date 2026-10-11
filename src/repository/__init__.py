"""Persistence layer: in-memory and file-system backends behind one interface."""

from src.repository.base import (
    ArtifactRepository,
    DatasetNotFoundError,
    DatasetRepository,
    RepositoryError,
)
from src.repository.factory import (
    get_artifact_repository,
    get_dataset_repository,
    reset_memory_repositories,
)
from src.repository.file_repository import (
    FileArtifactRepository,
    FileDatasetRepository,
)
from src.repository.memory_repository import (
    InMemoryArtifactRepository,
    InMemoryDatasetRepository,
)

__all__ = [
    "ArtifactRepository",
    "DatasetNotFoundError",
    "DatasetRepository",
    "FileArtifactRepository",
    "FileDatasetRepository",
    "InMemoryArtifactRepository",
    "InMemoryDatasetRepository",
    "RepositoryError",
    "get_artifact_repository",
    "get_dataset_repository",
    "reset_memory_repositories",
]
