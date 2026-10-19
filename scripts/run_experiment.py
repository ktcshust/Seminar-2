#!/usr/bin/env python3
"""Run the cross-validated comparison of every selected model.

Examples:
    # smoke run on synthetic data, 3 folds, everything kept in memory
    python scripts/run_experiment.py --synthetic --folds 3 --backend memory

    # the real experiment
    python scripts/run_experiment.py --models prior keyword logreg nbsvm linear_svm

All models share one fold assignment, which is what makes the paired
significance test at the end valid.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.dataio import (  # noqa: E402
    label_matrix,
    load_training_data,
    raw_files_available,
    stratified_subsample,
    texts,
)
from src.evaluation import (  # noqa: E402
    comparison_matrix,
    cross_validate_model,
    hierarchy_violations,
    identity_term_report,
    make_multilabel_folds,
)
from src.models import available_models, build_model  # noqa: E402
from src.repository import get_artifact_repository, get_dataset_repository  # noqa: E402

LOGGER = logging.getLogger("run_experiment")


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
    parser.add_argument("--folds", type=int, default=5, help="number of CV folds")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="evaluate on a stratified subsample of this size (for quick runs)",
    )
    parser.add_argument(
        "--backend",
        choices=["file", "memory"],
        default="file",
        help="persistence layer used for datasets and results",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="fall back to generated data when the Kaggle files are absent",
    )
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="score at a fixed 0.5 cut-off instead of tuning per label",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--tag", default="experiment", help="prefix for the stored result artifacts"
    )
    return parser.parse_args(argv)


def _run_metadata(
    args: argparse.Namespace, frame: pd.DataFrame, strategy: str, best: str
) -> dict[str, object]:
    """Record what actually ran, for the report's reproducibility subsection.

    The fold strategy in particular is decided at runtime -- iterative
    stratification is used when the optional package is installed and silently
    falls back to label powerset otherwise -- so it has to be read off the run
    rather than assumed.
    """
    import platform

    versions = {}
    for module in ("numpy", "pandas", "sklearn", "scipy", "joblib"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - an absent module is simply not reported
            versions[module] = "not installed"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": list(args.models),
        "best_model": best,
        "n_comments": int(len(frame)),
        "n_folds": int(args.folds),
        "fold_strategy": strategy,
        "seed": int(args.seed),
        # Whether the numbers may be quoted at all: a synthetic run is for
        # wiring checks only, and make_report_assets.py refuses to generate
        # report tables from one.
        "real_dataset": bool(raw_files_available()),
        "thresholds_tuned": not args.no_thresholds,
        "sample": args.sample,
        "persistence_backend": args.backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "libraries": versions,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )
    # Result tables are wide; the default terminal width truncates them.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    config = Config(persistence_backend=args.backend, random_state=args.seed)
    config.ensure_directories()
    datasets = get_dataset_repository(config)
    artifacts = get_artifact_repository(config)

    frame = load_training_data(
        config=config, repository=datasets, allow_synthetic=args.synthetic
    )
    if args.sample:
        frame = stratified_subsample(frame, args.sample, config=config)
    LOGGER.info("Evaluating on %d comments.", len(frame))

    comment_text = texts(frame, config).tolist()
    y = label_matrix(frame, config)
    label_names = list(config.label_columns)

    # One fold assignment shared by every model -> paired comparisons are valid.
    folds, strategy = make_multilabel_folds(
        y, n_splits=args.folds, random_state=args.seed
    )
    LOGGER.info("Fold strategy: %s (%d folds).", strategy, len(folds))

    results = {}
    for name in args.models:
        LOGGER.info("=== %s ===", name)
        result = cross_validate_model(
            model_factory=lambda name=name: build_model(name, label_names=label_names),
            texts=comment_text,
            y=y,
            label_names=label_names,
            folds=folds,
            strategy=strategy,
            random_state=args.seed,
            tune_thresholds=not args.no_thresholds,
        )
        results[name] = result
        artifacts.save(f"{args.tag}_{name}", result.to_dict())

    leaderboard = pd.DataFrame([result.summary for result in results.values()])
    leaderboard = leaderboard.sort_values("mean_column_auc_mean", ascending=False)

    print("\n" + "=" * 78)
    print("LEADERBOARD  (mean +/- std over folds)")
    print("=" * 78)
    print(
        leaderboard[
            [
                "mean_column_auc_mean",
                "mean_column_auc_std",
                "mean_column_pr_auc_mean",
                "macro_f1_mean",
                "micro_f1_mean",
                "subset_accuracy_mean",
                "fit_seconds_mean",
            ]
        ].round(4)
    )

    best = leaderboard.index[0]
    print("\n" + "=" * 78)
    print(f"PER-LABEL METRICS  ({best})")
    print("=" * 78)
    print(results[best].label_summary.round(4))

    if len(results) > 1:
        matrix = comparison_matrix(
            {name: result.fold_scores() for name, result in results.items()}
        )
        print("\n" + "=" * 78)
        print("PAIRWISE WILCOXON p-VALUES on mean column AUC")
        print(f"Bonferroni-corrected alpha = {matrix.attrs['bonferroni_alpha']:.4f}")
        print("=" * 78)
        print(matrix.round(4))
        # Persisted, not merely printed: the report's significance table
        # is generated from this file by scripts/make_report_assets.py.
        artifacts.save(
            f"{args.tag}_significance",
            {
                "metric": "mean_column_auc",
                "bonferroni_alpha": float(matrix.attrs["bonferroni_alpha"]),
                "models": list(matrix.index),
                "p_values": matrix.to_dict(),
            },
        )

    oof = results[best].oof_scores
    if oof is not None and not np.isnan(oof).all():
        # Audit at the operating point the rest of the report uses, not at a
        # hard-coded 0.5: the tuned toxic threshold sits well below it, so 0.5
        # would understate the false-positive rates this table is about.
        toxic_threshold = float(
            np.mean([threshold[0] for threshold in results[best].fold_thresholds])
        )
        bias = identity_term_report(
            comment_text, y, oof, label_index=0, threshold=toxic_threshold
        )
        if not bias.empty:
            print("\n" + "=" * 78)
            print(f"UNINTENDED BIAS on 'toxic'  ({best}, oof, thr={toxic_threshold:.3f})")
            print("=" * 78)
            print(bias.round(4))
            artifacts.save(f"{args.tag}_bias", bias.reset_index().to_dict("records"))

    # Out-of-fold scores are the input to the threshold sweep, the confusion
    # matrices and the error taxonomy.  They are saved as bare arrays on
    # purpose: FileArtifactRepository.save() serialises dicts and lists with
    # json.dumps(default=str), which would stringify a nested ndarray into a
    # truncated repr instead of refusing it.  A bare array takes the joblib
    # path and round-trips exactly.
    oof = results[best].oof_scores
    if oof is not None:
        artifacts.save(f"{args.tag}_oof_scores", oof)
    if results[best].oof_predictions is not None:
        artifacts.save(f"{args.tag}_oof_predictions", results[best].oof_predictions)

    predictions = results[best].oof_predictions
    if predictions is not None and (predictions >= 0).all():
        violations = hierarchy_violations(predictions, label_names)
        print("\n" + "=" * 78)
        print(f"LABEL-HIERARCHY VIOLATIONS  ({best}, child without toxic)")
        print("=" * 78)
        print(violations)
        artifacts.save(
            f"{args.tag}_hierarchy", violations.reset_index().to_dict("records")
        )

    artifacts.save(
        f"{args.tag}_leaderboard",
        leaderboard.reset_index(names="model").to_dict("records"),
    )
    artifacts.save(
        f"{args.tag}_metadata", _run_metadata(args, frame, strategy, best)
    )
    LOGGER.info(
        "Stored %d artifacts via the %s backend.",
        len(artifacts.list_artifacts()),
        config.persistence_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
