#!/usr/bin/env python3
"""Fold the models of a second experiment run into a base run.

The transformer needs a GPU environment that the linear models do not, so it is
evaluated in a separate ``run_experiment.py`` invocation.  Both runs use the same
seed and the same corpus, hence the same fold assignment, which keeps the
paired comparison between their per-fold scores valid.  This script rebuilds
the base run's leaderboard, significance matrix and metadata over the union of
models, so that ``make_report_assets.py`` sees one experiment.

Usage:
    python scripts/run_experiment.py --models distilbert --tag dl
    python scripts/merge_runs.py --base experiment --extra dl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.evaluation import comparison_matrix  # noqa: E402
from src.repository import get_artifact_repository  # noqa: E402

LOGGER = logging.getLogger("merge_runs")

#: Artifacts that describe the best model only, and move with it.
BEST_MODEL_ARTIFACTS = ("bias", "hierarchy", "oof_scores", "oof_predictions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="experiment", help="tag merged into")
    parser.add_argument("--extra", required=True, help="tag whose models are added")
    parser.add_argument(
        "--backend",
        choices=["file", "memory"],
        default="file",
        help="persistence layer to read and write the artifacts through",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    artifacts = get_artifact_repository(Config(persistence_backend=args.backend))

    base_meta = artifacts.load(f"{args.base}_metadata")
    extra_meta = artifacts.load(f"{args.extra}_metadata")

    # Identical folds are what make the merged paired test legitimate; refuse
    # rather than silently compare scores computed on different splits.
    for key in ("n_comments", "n_folds", "seed", "fold_strategy", "sample"):
        if base_meta[key] != extra_meta[key]:
            LOGGER.error(
                "Runs disagree on %s (%r vs %r); their folds differ.",
                key,
                base_meta[key],
                extra_meta[key],
            )
            return 1

    extra_models = [m for m in extra_meta["models"] if m not in base_meta["models"]]
    for name in extra_models:
        artifacts.save(f"{args.base}_{name}", artifacts.load(f"{args.extra}_{name}"))
    models = list(base_meta["models"]) + extra_models
    results = {name: artifacts.load(f"{args.base}_{name}") for name in models}

    leaderboard = pd.DataFrame(
        [{"model": name, **result["summary"]} for name, result in results.items()]
    ).sort_values("mean_column_auc_mean", ascending=False)
    best = str(leaderboard.iloc[0]["model"])

    matrix = comparison_matrix(
        {
            name: np.array([fold["mean_column_auc"] for fold in result["fold_metrics"]])
            for name, result in results.items()
        }
    )
    artifacts.save(
        f"{args.base}_significance",
        {
            "metric": "mean_column_auc",
            "bonferroni_alpha": float(matrix.attrs["bonferroni_alpha"]),
            "models": list(matrix.index),
            "p_values": matrix.to_dict(),
        },
    )

    if best in extra_models:
        for kind in BEST_MODEL_ARTIFACTS:
            if artifacts.exists(f"{args.extra}_{kind}"):
                artifacts.save(
                    f"{args.base}_{kind}", artifacts.load(f"{args.extra}_{kind}")
                )

    artifacts.save(f"{args.base}_leaderboard", leaderboard.to_dict("records"))
    merged = [
        {key: extra_meta[key] for key in ("timestamp", "python", "libraries")}
        | {"tag": args.extra, "models": extra_models}
    ]
    artifacts.save(
        f"{args.base}_metadata",
        base_meta
        | {
            "models": models,
            "best_model": best,
            "merged_runs": base_meta.get("merged_runs", []) + merged,
        },
    )

    print(
        leaderboard[["model", "mean_column_auc_mean", "macro_f1_mean"]]
        .round(4)
        .to_string(index=False)
    )
    LOGGER.info("Merged %s into '%s'; best model is %s.", extra_models, args.base, best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
