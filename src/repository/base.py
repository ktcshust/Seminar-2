"""Abstract persistence layer.

The course requires that data can be handled *both* in memory and through files
on the file system, and that swapping one for the other leaves the business
logic untouched.  That is achieved with the two abstract base classes below:

* :class:`DatasetRepository` stores tabular data (``pandas.DataFrame``);
* :class:`ArtifactRepository` stores arbitrary Python objects, e.g. a fitted
  classifier or a dictionary of metrics.

Every consumer in :mod:`src` type-hints against these interfaces only, never
against a concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class RepositoryError(RuntimeError):
    """Raised when a repository cannot honour a request."""


class DatasetNotFoundError(RepositoryError, KeyError):
    """Raised when reading a dataset that was never stored."""


class DatasetRepository(ABC):
    """Store and retrieve named tabular datasets."""

    @abstractmethod
    def save(self, name: str, frame: pd.DataFrame) -> None:
        """Persist ``frame`` under ``name``, overwriting any previous value."""

    @abstractmethod
    def load(self, name: str) -> pd.DataFrame:
        """Return the dataset stored under ``name``.

        Raises:
            DatasetNotFoundError: if ``name`` is unknown.
        """

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Return whether a dataset called ``name`` is available."""

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """Return every available dataset name, sorted alphabetically."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Remove ``name``.  Deleting an unknown dataset is a no-op."""

    # -- conveniences shared by every implementation ----------------------- #

    def load_or_compute(self, name: str, factory) -> pd.DataFrame:
        """Return a cached dataset, computing and storing it on a cache miss.

        Args:
            name: dataset identifier.
            factory: zero-argument callable returning a ``DataFrame``.
        """
        if self.exists(name):
            return self.load(name)
        frame = factory()
        self.save(name, frame)
        return frame

    def __contains__(self, name: object) -> bool:
        """Support ``name in repository``."""
        return isinstance(name, str) and self.exists(name)

    def __len__(self) -> int:
        """Return the number of stored datasets."""
        return len(self.list_datasets())


class ArtifactRepository(ABC):
    """Store and retrieve arbitrary Python objects (models, metric dicts...)."""

    @abstractmethod
    def save(self, name: str, obj: Any) -> None:
        """Persist ``obj`` under ``name``, overwriting any previous value."""

    @abstractmethod
    def load(self, name: str) -> Any:
        """Return the artifact stored under ``name``."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Return whether an artifact called ``name`` is available."""

    @abstractmethod
    def list_artifacts(self) -> list[str]:
        """Return every available artifact name, sorted alphabetically."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Remove ``name``.  Deleting an unknown artifact is a no-op."""

    def __contains__(self, name: object) -> bool:
        """Support ``name in repository``."""
        return isinstance(name, str) and self.exists(name)
