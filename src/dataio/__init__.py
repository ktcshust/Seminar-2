"""Dataset acquisition, validation and caching."""

from src.dataio.loader import (
    SYNTHETIC_TRAIN_DATASET,
    TEST_DATASET,
    TRAIN_DATASET,
    load_scored_test_data,
    load_training_data,
    raw_files_available,
    stratified_subsample,
)
from src.dataio.schema import SchemaError, label_matrix, texts, validate_labelled_frame
from src.dataio.synthetic import generate_synthetic_dataset

__all__ = [
    "SYNTHETIC_TRAIN_DATASET",
    "SchemaError",
    "TEST_DATASET",
    "TRAIN_DATASET",
    "generate_synthetic_dataset",
    "label_matrix",
    "load_scored_test_data",
    "load_training_data",
    "raw_files_available",
    "stratified_subsample",
    "texts",
    "validate_labelled_frame",
]
