"""File-system implementations of the persistence layer.

Datasets are written as CSV (default, human-inspectable and what Kaggle ships)
or Parquet (faster and type-preserving on large frames).  Artifacts are written
with ``joblib`` for estimators and with JSON for plain dictionaries, so that
metric files stay readable and can be diffed in Git.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.repository.base import (
    ArtifactRepository,
    DatasetNotFoundError,
    DatasetRepository,
    RepositoryError,
)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate(name: str) -> str:
    """Reject names that would escape the repository directory."""
    if not _SAFE_NAME.match(name):
        raise RepositoryError(
            f"Invalid name {name!r}: use letters, digits, '.', '_' or '-' only."
        )
    return name


class FileDatasetRepository(DatasetRepository):
    """Persist datasets under ``root`` as CSV or Parquet files."""

    def __init__(self, root: Path | str, file_format: str = "csv") -> None:
        if file_format not in {"csv", "parquet"}:
            raise RepositoryError(f"Unsupported file format {file_format!r}.")
        self.root = Path(root)
        self.file_format = file_format
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def _suffix(self) -> str:
        return ".csv" if self.file_format == "csv" else ".parquet"

    def _path(self, name: str) -> Path:
        return self.root / f"{_validate(name)}{self._suffix}"

    def save(self, name: str, frame: pd.DataFrame) -> None:
        """Write ``frame`` to ``<root>/<name>.<csv|parquet>``."""
        path = self._path(name)
        if self.file_format == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)

    def load(self, name: str) -> pd.DataFrame:
        """Read the frame back from disk."""
        path = self._path(name)
        if not path.exists():
            raise DatasetNotFoundError(
                f"No dataset named {name!r} in {self.root}; "
                f"available: {self.list_datasets()}"
            )
        if self.file_format == "csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def exists(self, name: str) -> bool:
        """Return whether the backing file is present."""
        return self._path(name).exists()

    def list_datasets(self) -> list[str]:
        """Return the names of the dataset files in ``root``, sorted."""
        return sorted(p.stem for p in self.root.glob(f"*{self._suffix}"))

    def delete(self, name: str) -> None:
        """Remove the backing file if it exists."""
        self._path(name).unlink(missing_ok=True)


class FileArtifactRepository(ArtifactRepository):
    """Persist models (``.joblib``) and metric dictionaries (``.json``)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _json_path(self, name: str) -> Path:
        return self.root / f"{_validate(name)}.json"

    def _joblib_path(self, name: str) -> Path:
        return self.root / f"{_validate(name)}.joblib"

    def save(self, name: str, obj: Any) -> None:
        """Write ``obj`` as JSON when it round-trips, otherwise with joblib."""
        # Prefer JSON when the object round-trips through it: the resulting file
        # is readable in a pull request, which matters for the metric tables.
        if isinstance(obj, (dict, list)):
            try:
                payload = json.dumps(obj, indent=2, sort_keys=True, default=str)
            except TypeError:
                pass
            else:
                self._json_path(name).write_text(payload, encoding="utf-8")
                self._joblib_path(name).unlink(missing_ok=True)
                return
        joblib.dump(obj, self._joblib_path(name))
        self._json_path(name).unlink(missing_ok=True)

    def load(self, name: str) -> Any:
        """Read the artifact back, whichever format it was written in."""
        json_path = self._json_path(name)
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        joblib_path = self._joblib_path(name)
        if joblib_path.exists():
            return joblib.load(joblib_path)
        raise DatasetNotFoundError(
            f"No artifact named {name!r} in {self.root}; "
            f"available: {self.list_artifacts()}"
        )

    def exists(self, name: str) -> bool:
        """Return whether either backing file is present."""
        return self._json_path(name).exists() or self._joblib_path(name).exists()

    def list_artifacts(self) -> list[str]:
        """Return the artifact names found in ``root``, sorted."""
        names = {p.stem for p in self.root.glob("*.json")}
        names |= {p.stem for p in self.root.glob("*.joblib")}
        return sorted(names)

    def delete(self, name: str) -> None:
        """Remove both possible backing files."""
        self._json_path(name).unlink(missing_ok=True)
        self._joblib_path(name).unlink(missing_ok=True)
