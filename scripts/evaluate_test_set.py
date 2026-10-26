#!/usr/bin/env python3
"""Score models on the scored part of Kaggle's official test set.

Cross validation answers "which model is better"; this script answers "how
would it have fared in the competition".  Each model is trained exactly as in
one cross-validation fold -- fitted on 85% of the full training set, with the
per-label thresholds tuned on the remaining 15% -- and then scored once on the
63,978 test comments that Kaggle labelled after the competition closed.  The
mean column-wise AUC is therefore directly comparable with the private
leaderboard.

Usage:
    python scripts/evaluate_test_set.py
    python scripts/evaluate_test_set.py --models logreg nbsvm
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.dataio import (  # noqa: E402
    label_matrix,
    load_scored_test_data,
    load_training_data,
    texts,
)
from src.evaluation import (  # noqa: E402
    multilabel_train_test_split,
    optimise_thresholds,
    overall_metrics,
    per_label_metrics,
)
from src.models import available_models, build_model  # noqa: E402
from src.repository import get_artifact_repository, get_dataset_repository  # noqa: E402

LOGGER = logging.getLogger("evaluate_test_set")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["prior", "keyword", "logreg", "nbsvm", "linear_svm"],
        choices=available_models(),
        help="models to evaluate",
    )
    parser.add_argument(
        "--backend",
        choices=["file", "memory"],
        default="file",
        help="persistence layer used for datasets and results",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--tag", default="test", help="prefix for the stored result artifacts"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )

    config = Config(persistence_backend=args.backend, random_state=args.seed)
    datasets = get_dataset_repository(config)
    artifacts = get_artifact_repository(config)

    train = load_training_data(config=config, repository=datasets)
    test = load_scored_test_data(config=config, repository=datasets)
    LOGGER.info("Training on %d comments, testing on %d.", len(train), len(test))

    train_text = texts(train, config).tolist()
    test_text = texts(test, config).tolist()
    y_train = label_matrix(train, config)
    y_test = label_matrix(test, config)
    label_names = list(config.label_columns)

    # The same split rule as inside a cross-validation fold, so the numbers on
    # the test set and the cross-validated ones describe the same procedure.
    fit_index, tune_index = multilabel_train_test_split(
        y_train, test_size=0.15, random_state=args.seed
    )

    rows = []
    for name in args.models:
        LOGGER.info("=== %s ===", name)
        model = build_model(name, label_names=label_names)
        started = time.perf_counter()
        model.fit([train_text[i] for i in fit_index], y_train[fit_index])
        elapsed = time.perf_counter() - started

        thresholds = optimise_thresholds(
            y_train[tune_index],
            model.predict_proba([train_text[i] for i in tune_index]),
        )
        scores = model.predict_proba(test_text)

        summary = overall_metrics(y_test, scores, label_names, thresholds)
        per_label = per_label_metrics(y_test, scores, label_names, thresholds)
        artifacts.save(
            f"{args.tag}_{name}",
            {
                "model_name": name,
                "n_train": len(fit_index),
                "n_test": len(test),
                "fit_seconds": elapsed,
                "thresholds": thresholds.tolist(),
                "summary": summary,
                "label_summary": per_label.to_dict(),
            },
        )
        rows.append({"model": name, **summary, "fit_seconds": elapsed})
        LOGGER.info(
            "[%s] test AUC=%.4f  macro-F1=%.4f  (%.1fs)",
            name,
            summary["mean_column_auc"],
            summary["macro_f1"],
            elapsed,
        )

    leaderboard = pd.DataFrame(rows).sort_values("mean_column_auc", ascending=False)
    artifacts.save(f"{args.tag}_leaderboard", leaderboard.to_dict("records"))
    print(
        leaderboard[
            ["model", "mean_column_auc", "mean_column_pr_auc", "macro_f1", "micro_f1"]
        ]
        .round(4)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
