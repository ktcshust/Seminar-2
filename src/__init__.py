"""Toxic Comment Classification - AI4SE course project.

The package is organised in layers so that the *business logic* (preprocessing,
models, evaluation) never depends on how the data is physically stored:

    src.repository   persistence abstraction (in-memory  <-> file system)
    src.dataio       dataset acquisition and schema validation
    src.preprocessing text cleaning and vectorisation
    src.models       classifiers, all exposing the same fit/predict_proba API
    src.evaluation   metrics, threshold tuning, cross validation
"""

__version__ = "1.0.0"
