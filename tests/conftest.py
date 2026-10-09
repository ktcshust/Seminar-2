"""Shared fixtures.

Adding the project root to ``sys.path`` lets the tests run with a plain
``pytest`` invocation, without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.dataio import generate_synthetic_dataset  # noqa: E402
from src.repository import reset_memory_repositories  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_memory_repositories():
    """Keep the in-memory singletons from leaking between test cases."""
    reset_memory_repositories()
    yield
    reset_memory_repositories()


@pytest.fixture
def sample_frame():
    """A small synthetic dataset with every label represented."""
    return generate_synthetic_dataset(n_samples=600, random_state=0)


@pytest.fixture
def file_config(tmp_path):
    """A configuration whose file backend writes inside ``tmp_path``."""
    return Config(
        persistence_backend="file",
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )


@pytest.fixture
def memory_config(tmp_path):
    """The same configuration, but backed by memory."""
    return Config(
        persistence_backend="memory",
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )
