"""Schema checks for the Jigsaw dataset.

Catching a malformed frame here is much cheaper than discovering it after a
forty-minute cross-validation run.
"""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG, Config


class SchemaError(ValueError):
    """Raised when a frame does not look like the Jigsaw dataset."""


def validate_labelled_frame(frame: pd.DataFrame, config: Config | None = None) -> None:
    """Check that ``frame`` carries the text column and the six binary labels.

    Raises:
        SchemaError: on a missing column, a non-binary label or a null text.
    """
    config = CONFIG if config is None else config
    required = [config.text_column, *config.label_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing column(s): {missing}")

    if frame[config.text_column].isna().any():
        raise SchemaError(f"Column {config.text_column!r} contains null values.")

    for label in config.label_columns:
        values = set(pd.unique(frame[label].dropna()))
        if not values <= {0, 1}:
            raise SchemaError(
                f"Label {label!r} is not binary; found values {sorted(values)}."
            )


def label_matrix(frame: pd.DataFrame, config: Config | None = None):
    """Return the ``(n_samples, 6)`` integer label matrix of ``frame``."""
    config = CONFIG if config is None else config
    return frame[list(config.label_columns)].to_numpy(dtype=int)


def texts(frame: pd.DataFrame, config: Config | None = None) -> pd.Series:
    """Return the raw comment text column of ``frame``."""
    config = CONFIG if config is None else config
    return frame[config.text_column].astype(str)
