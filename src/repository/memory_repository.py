"""In-memory implementations of the persistence layer.

Everything lives in a dictionary owned by the repository instance, which makes
these classes the natural choice for unit tests and for notebook sessions where
intermediate results should not touch the disk.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from src.repository.base import (
    ArtifactRepository,
    DatasetNotFoundError,
    DatasetRepository,
)


class InMemoryDatasetRepository(DatasetRepository):
    """Keep every dataset in a private dictionary.

    Frames are copied on the way in and on the way out so that a caller cannot
    mutate stored data by accident -- the file backend has the same semantics,
    and the two must be interchangeable.
    """

    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    def save(self, name: str, frame: pd.DataFrame) -> None:
        """Store a deep copy of ``frame`` under ``name``."""
        self._store[name] = frame.copy(deep=True)

    def load(self, name: str) -> pd.DataFrame:
        """Return a deep copy of the stored frame."""
        try:
            return self._store[name].copy(deep=True)
        except KeyError as exc:
            raise DatasetNotFoundError(
                f"No dataset named {name!r}; available: {self.list_datasets()}"
            ) from exc

    def exists(self, name: str) -> bool:
        """Return whether ``name`` is in the dictionary."""
        return name in self._store

    def list_datasets(self) -> list[str]:
        """Return the stored dataset names, sorted."""
        return sorted(self._store)

    def delete(self, name: str) -> None:
        """Drop ``name`` if present."""
        self._store.pop(name, None)

    def clear(self) -> None:
        """Drop every stored dataset (handy between test cases)."""
        self._store.clear()


class InMemoryArtifactRepository(ArtifactRepository):
    """Keep fitted models and metric dictionaries in memory."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def save(self, name: str, obj: Any) -> None:
        """Store ``obj`` under ``name`` by reference."""
        self._store[name] = obj

    def load(self, name: str) -> Any:
        """Return the stored artifact, copying plain containers."""
        try:
            obj = self._store[name]
        except KeyError as exc:
            raise DatasetNotFoundError(
                f"No artifact named {name!r}; available: {self.list_artifacts()}"
            ) from exc
        # Plain containers are copied, fitted estimators are returned as-is
        # (deep-copying a large model on every read would be wasteful).
        return copy.deepcopy(obj) if isinstance(obj, (dict, list)) else obj

    def exists(self, name: str) -> bool:
        """Return whether ``name`` is in the dictionary."""
        return name in self._store

    def list_artifacts(self) -> list[str]:
        """Return the stored artifact names, sorted."""
        return sorted(self._store)

    def delete(self, name: str) -> None:
        """Drop ``name`` if present."""
        self._store.pop(name, None)

    def clear(self) -> None:
        """Drop every stored artifact."""
        self._store.clear()
