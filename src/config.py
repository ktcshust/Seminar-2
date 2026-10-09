"""Central configuration.

This is the *only* module that decides which persistence layer is used.  The
non-functional requirement of the course states that switching between the
in-memory and the file-system backend must involve as few changes as possible:
here it is a single value (or the ``TCC_PERSISTENCE`` environment variable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #

#: The six binary targets of the Jigsaw challenge, in the canonical order used
#: by the competition's submission format.
LABEL_COLUMNS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)

TEXT_COLUMN = "comment_text"
ID_COLUMN = "id"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    """Runtime configuration of the application."""

    # -- persistence ------------------------------------------------------- #
    #: Either ``"file"`` or ``"memory"``.  Nothing else in the code base reads
    #: this value directly; the repository factory does.
    persistence_backend: str = os.environ.get("TCC_PERSISTENCE", "file")

    #: Serialisation format used by the file backend ("csv" or "parquet").
    file_format: str = os.environ.get("TCC_FILE_FORMAT", "csv")

    # -- paths ------------------------------------------------------------- #
    project_root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    results_dir: Path = PROJECT_ROOT / "results"
    figures_dir: Path = PROJECT_ROOT / "report" / "figures"

    # -- experiment -------------------------------------------------------- #
    random_state: int = 42
    n_splits: int = 5

    # -- schema ------------------------------------------------------------ #
    label_columns: tuple[str, ...] = field(default=LABEL_COLUMNS)
    text_column: str = TEXT_COLUMN
    id_column: str = ID_COLUMN

    def __post_init__(self) -> None:
        """Reject an unsupported persistence backend at construction time."""
        if self.persistence_backend not in {"file", "memory"}:
            raise ValueError(
                f"Unknown persistence backend {self.persistence_backend!r}; "
                "expected 'file' or 'memory'."
            )

    def ensure_directories(self) -> None:
        """Create every output directory the application writes to."""
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.results_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


#: Default configuration used by the scripts and notebooks.
CONFIG = Config()
