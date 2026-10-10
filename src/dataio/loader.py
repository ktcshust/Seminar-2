"""Acquiring the Jigsaw dataset and exposing it through the repository layer.

The raw CSV files are never read twice: :func:`load_training_data` caches the
validated frame through a :class:`~src.repository.base.DatasetRepository`, so
the same call works whether the project is configured to keep data in memory or
on disk.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import CONFIG, Config
from src.dataio.schema import validate_labelled_frame
from src.dataio.synthetic import generate_synthetic_dataset
from src.repository.base import DatasetRepository
from src.repository.factory import get_dataset_repository

LOGGER = logging.getLogger(__name__)

TRAIN_DATASET = "train"
TEST_DATASET = "test_scored"

#: Synthetic data is cached under its own name.  Sharing the cache key with the
#: real corpus would mean that a smoke run with --synthetic and the file backend
#: leaves a fake "train" behind, which every later run would silently reuse.
SYNTHETIC_TRAIN_DATASET = "train_synthetic"

DOWNLOAD_HINT = """
The Jigsaw files were not found in {raw_dir}.

Download them once.  A Kaggle account is required, and the competition rules
must be accepted on the website first (Rules tab -> "I Understand and Accept"),
otherwise the API answers 403 even with valid credentials.

    pip install kaggle
    kaggle auth login                  # OAuth; nothing to store locally

    # Or, without OAuth: generate a token at kaggle.com/settings/api, then
    #   mkdir -p ~/.kaggle && echo <TOKEN> > ~/.kaggle/access_token
    #   chmod 600 ~/.kaggle/access_token
    # A legacy ~/.kaggle/kaggle.json still works as a fallback.

    python scripts/download_data.py

Expected afterwards: train.csv, test.csv, test_labels.csv
"""


def raw_files_available(config: Config | None = None) -> bool:
    """Return whether ``data/raw/train.csv`` is present."""
    config = CONFIG if config is None else config
    return (config.raw_dir / "train.csv").exists()


def load_training_data(
    config: Config | None = None,
    repository: DatasetRepository | None = None,
    allow_synthetic: bool = False,
    n_synthetic: int = 8_000,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Return the labelled training frame, caching it in the repository.

    Args:
        config: configuration to use; defaults to :data:`src.config.CONFIG`.
        repository: where to cache the frame; defaults to the configured one.
        allow_synthetic: fall back to :mod:`src.dataio.synthetic` when the real
            CSV files are missing.  Intended for tests and smoke runs only; the
            result is cached under a separate name so it can never be mistaken
            for the real corpus later.
        n_synthetic: size of the synthetic fallback.
        force_reload: ignore anything already cached.

    Raises:
        FileNotFoundError: when the raw files are missing and ``allow_synthetic``
            is ``False``.
    """
    config = CONFIG if config is None else config
    # `is None` rather than `or`: a repository defines __len__, so an empty
    # one is falsy and `or` would silently discard the caller's instance.
    repository = get_dataset_repository(config) if repository is None else repository

    train_path = config.raw_dir / "train.csv"
    use_synthetic = not train_path.exists()
    if use_synthetic and not allow_synthetic:
        raise FileNotFoundError(DOWNLOAD_HINT.format(raw_dir=config.raw_dir))

    name = SYNTHETIC_TRAIN_DATASET if use_synthetic else TRAIN_DATASET
    if not force_reload and repository.exists(name):
        return repository.load(name)

    if use_synthetic:
        LOGGER.warning(
            "Raw dataset missing - generating %d SYNTHETIC rows. "
            "Results from this data must not be reported.",
            n_synthetic,
        )
        frame = generate_synthetic_dataset(n_synthetic, config=config)
    else:
        LOGGER.info("Reading %s", train_path)
        frame = pd.read_csv(train_path)

    frame[config.text_column] = frame[config.text_column].fillna("").astype(str)
    validate_labelled_frame(frame, config=config)
    repository.save(name, frame)
    return frame


def load_scored_test_data(
    config: Config | None = None,
    repository: DatasetRepository | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Return the *scored* part of the official test set.

    ``test_labels.csv`` marks the rows that Kaggle did not score with ``-1`` in
    every label column.  Keeping them would silently corrupt every metric, so
    they are dropped here.

    Raises:
        FileNotFoundError: when ``test.csv`` or ``test_labels.csv`` is missing.
    """
    config = CONFIG if config is None else config
    # `is None` rather than `or`: a repository defines __len__, so an empty
    # one is falsy and `or` would silently discard the caller's instance.
    repository = get_dataset_repository(config) if repository is None else repository

    if not force_reload and repository.exists(TEST_DATASET):
        return repository.load(TEST_DATASET)

    test_path = config.raw_dir / "test.csv"
    labels_path = config.raw_dir / "test_labels.csv"
    if not (test_path.exists() and labels_path.exists()):
        raise FileNotFoundError(DOWNLOAD_HINT.format(raw_dir=config.raw_dir))

    test = pd.read_csv(test_path)
    labels = pd.read_csv(labels_path)
    frame = test.merge(labels, on=config.id_column, how="inner")

    unscored = (frame[list(config.label_columns)] == -1).any(axis=1)
    LOGGER.info(
        "Dropping %d unscored rows out of %d in the official test set.",
        int(unscored.sum()),
        len(frame),
    )
    frame = frame.loc[~unscored].reset_index(drop=True)

    frame[config.text_column] = frame[config.text_column].fillna("").astype(str)
    validate_labelled_frame(frame, config=config)
    repository.save(TEST_DATASET, frame)
    return frame


def stratified_subsample(
    frame: pd.DataFrame,
    n_samples: int,
    config: Config | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return ``n_samples`` rows keeping the ratio of clean to flagged comments.

    Sampling uniformly at random would leave almost no ``threat`` examples in a
    small subset, which makes quick experiments misleading.
    """
    config = CONFIG if config is None else config
    if n_samples >= len(frame):
        return frame.reset_index(drop=True)

    seed = config.random_state if random_state is None else random_state
    flagged = frame[list(config.label_columns)].sum(axis=1) > 0
    positives, negatives = frame[flagged], frame[~flagged]

    n_positive = min(len(positives), round(n_samples * len(positives) / len(frame)))
    n_negative = min(len(negatives), n_samples - n_positive)

    sample = pd.concat(
        [
            positives.sample(n=n_positive, random_state=seed),
            negatives.sample(n=n_negative, random_state=seed),
        ]
    )
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
