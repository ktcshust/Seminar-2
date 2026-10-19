#!/usr/bin/env python3
"""Render every figure the report includes, from one stored experiment run.

The notebooks each re-run their own cross validation, and notebook 02 does so on
a 20,000-comment subsample by default.  Figures produced that way disagree with
the tables, which come from the full run -- a reader comparing
Figure "leaderboard" with Table "leaderboard" would see two different numbers
for the same quantity.

This script removes that failure mode: it reads the artifacts written by
``run_experiment.py`` and the raw corpus, and renders all nine figures without
fitting anything.  The notebooks stay what they are good for -- exploring; the
report's figures come from here.

Usage:
    python scripts/make_report_figures.py            # tag "experiment"
    python scripts/make_report_figures.py --tag ablation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a terminal session

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.dataio import (  # noqa: E402
    label_matrix,
    load_training_data,
    stratified_subsample,
)
from src.evaluation.metrics import per_label_metrics  # noqa: E402
from src.repository import get_artifact_repository, get_dataset_repository  # noqa: E402
from src.repository.base import DatasetNotFoundError  # noqa: E402

LOGGER = logging.getLogger("make_report_figures")

PRIMARY = "#4C72B0"
ACCENT = "#C44E52"


def save(figure: plt.Figure, path: Path, name: str) -> Path:
    """Write one figure at the resolution the report expects and close it."""
    target = path / f"{name}.png"
    figure.tight_layout()
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return target


# --------------------------------------------------------------------------- #
# Corpus figures -- these need only the raw data
# --------------------------------------------------------------------------- #


def figure_label_frequency(frame: pd.DataFrame, labels: list[str]) -> plt.Figure:
    """Positive comments per label."""
    counts = frame[labels].sum().sort_values(ascending=False)
    figure, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color=PRIMARY)
    ax.set(xlabel="positive comments", ylabel="", title="Label frequency")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)
    sns.despine(ax=ax)
    return figure


def figure_label_cooccurrence(frame: pd.DataFrame, labels: list[str]) -> plt.Figure:
    """Conditional co-occurrence P(column | row)."""
    cooccurrence = frame[labels].T.dot(frame[labels])
    conditional = cooccurrence.div(np.diag(cooccurrence), axis=0)
    figure, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        conditional,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "P(column | row)"},
        ax=ax,
    )
    ax.set_title("Conditional label co-occurrence")
    return figure


def figure_comment_length(frame: pd.DataFrame, config: Config) -> plt.Figure:
    """Length distribution, with the transformer's truncation point marked."""
    words = frame[config.text_column].str.split().str.len()
    toxic = frame["toxic"] == 1

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    sns.histplot(words.clip(upper=400), bins=60, ax=axes[0], color=PRIMARY)
    axes[0].axvline(192, color=ACCENT, ls="--", lw=1.2)
    axes[0].text(200, axes[0].get_ylim()[1] * 0.8, "192 tokens", color=ACCENT, fontsize=8)
    axes[0].set(title="Comment length (words, clipped at 400)", xlabel="words")

    comparison = pd.DataFrame(
        {
            "toxic": words[toxic].clip(upper=400),
            "clean": words[~toxic].clip(upper=400),
        }
    )
    sns.boxplot(data=comparison, ax=axes[1], palette=[ACCENT, PRIMARY], showfliers=False)
    axes[1].set(title="Length by toxicity", ylabel="words")
    for ax in axes:
        sns.despine(ax=ax)
    return figure


# --------------------------------------------------------------------------- #
# Result figures -- these read the stored artifacts
# --------------------------------------------------------------------------- #


def figure_leaderboard(leaderboard: list[dict]) -> plt.Figure:
    """Mean column-wise AUC per model, with the spread across folds."""
    frame = pd.DataFrame(leaderboard).set_index("model")
    frame = frame.sort_values("mean_column_auc_mean")
    figure, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.barh(
        frame.index,
        frame["mean_column_auc_mean"],
        xerr=frame["mean_column_auc_std"],
        color=PRIMARY,
        capsize=4,
    )
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.text(0.505, -0.4, "chance", color="grey", fontsize=8)
    ax.set(xlabel="mean column-wise ROC-AUC", xlim=(0.45, 1.0))
    sns.despine(ax=ax)
    return figure


def figure_pr_auc_per_label(
    per_model: dict[str, dict], labels: list[str]
) -> plt.Figure:
    """PR-AUC per label, one bar group per model."""
    frame = pd.DataFrame(
        {
            name: pd.DataFrame(result["label_summary"]).loc[labels, "pr_auc"]
            for name, result in per_model.items()
        }
    )
    figure, ax = plt.subplots(figsize=(8, 3.5))
    frame.plot.bar(ax=ax, width=0.8)
    ax.set(ylabel="PR-AUC", xlabel="", title="Average precision per label")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    sns.despine(ax=ax)
    return figure


def figure_cost_vs_accuracy(leaderboard: list[dict]) -> plt.Figure:
    """Accuracy against the time it costs to fit."""
    frame = pd.DataFrame(leaderboard).set_index("model")
    # A trivial model can fit in microseconds; clip so the log axis stays sane.
    seconds = frame["fit_seconds_mean"].clip(lower=1e-2)
    figure, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(seconds, frame["mean_column_auc_mean"], s=70, color=PRIMARY)

    # The good models cluster in one corner -- near-identical AUC at near-
    # identical cost -- so a fixed label offset stacks them into an unreadable
    # blob.  Group the points that share a neighbourhood and fan their labels
    # out vertically, with a leader line back to the marker.
    positions = [
        (name, np.log10(max(row["fit_seconds_mean"], 1e-2)), row["mean_column_auc_mean"])
        for name, row in frame.iterrows()
    ]
    span = max(p[2] for p in positions) - min(p[2] for p in positions) or 1.0
    groups: list[list[tuple[str, float, float]]] = []
    for point in sorted(positions, key=lambda p: -p[2]):
        for group in groups:
            if abs(group[0][1] - point[1]) < 0.5 and abs(group[0][2] - point[2]) < (
                0.05 * span
            ):
                group.append(point)
                break
        else:
            groups.append([point])

    low = min(p[1] for p in positions)
    high = max(p[1] for p in positions)
    for group in groups:
        # Labels on a point in the right-hand half would run off the axes, so
        # flip them to the left of their marker.
        flip = (group[0][1] - low) > 0.6 * (high - low or 1.0)
        for rank, (name, _, _) in enumerate(group):
            row = frame.loc[name]
            ax.annotate(
                name,
                (max(row["fit_seconds_mean"], 1e-2), row["mean_column_auc_mean"]),
                textcoords="offset points",
                xytext=(-10 if flip else 10, 6 - rank * 12 if len(group) > 1 else 6),
                ha="right" if flip else "left",
                fontsize=9,
                arrowprops=(
                    {"arrowstyle": "-", "lw": 0.5, "color": "0.6"}
                    if len(group) > 1
                    else None
                ),
            )
    ax.margins(x=0.15, y=0.12)
    ax.set(
        xscale="log",
        xlabel="mean fit time per fold (s, log scale)",
        ylabel="mean column-wise AUC",
        title="Accuracy against training cost",
    )
    sns.despine(ax=ax)
    return figure


def figure_threshold_sweep(
    y_true: np.ndarray,
    oof: np.ndarray,
    labels: list[str],
    tuned: np.ndarray,
) -> plt.Figure:
    """F1 against the decision threshold, per label, with the tuned point."""
    grid = np.linspace(0.05, 0.95, 37)
    figure, ax = plt.subplots(figsize=(7.5, 4))
    palette = sns.color_palette("deep", len(labels))

    for index, label in enumerate(labels):
        truth = y_true[:, index]
        scores = oof[:, index]
        curve = []
        for threshold in grid:
            predicted = (scores >= threshold).astype(int)
            true_positive = float(np.sum((predicted == 1) & (truth == 1)))
            precision_denominator = predicted.sum()
            recall_denominator = truth.sum()
            if not precision_denominator or not recall_denominator:
                curve.append(0.0)
                continue
            precision = true_positive / precision_denominator
            recall = true_positive / recall_denominator
            total = precision + recall
            curve.append(0.0 if total == 0 else 2 * precision * recall / total)
        ax.plot(grid, curve, label=label, color=palette[index])
        ax.axvline(tuned[index], color=palette[index], ls=":", lw=1, alpha=0.7)

    ax.set(
        xlabel="decision threshold",
        ylabel="$F_1$",
        title="$F_1$ against threshold (out-of-fold); dotted = tuned",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
    sns.despine(ax=ax)
    return figure


def figure_confusion_matrices(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> plt.Figure:
    """One 2x2 confusion matrix per label, at the tuned thresholds."""
    figure, axes = plt.subplots(2, 3, figsize=(11, 6))
    for index, (label, ax) in enumerate(zip(labels, axes.ravel(), strict=False)):
        truth = y_true[:, index]
        predicted = y_pred[:, index]
        matrix = np.array(
            [
                [
                    int(np.sum((truth == 0) & (predicted == 0))),
                    int(np.sum((truth == 0) & (predicted == 1))),
                ],
                [
                    int(np.sum((truth == 1) & (predicted == 0))),
                    int(np.sum((truth == 1) & (predicted == 1))),
                ],
            ]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["pred 0", "pred 1"],
            yticklabels=["true 0", "true 1"],
        )
        ax.set_title(label, fontsize=10)
    figure.suptitle("Per-label confusion at the tuned thresholds (out-of-fold)")
    return figure


def figure_identity_bias(bias: list[dict]) -> plt.Figure:
    """Subgroup against background false-positive rate, per identity term.

    Each term has its own background rate -- it is measured on the comments that
    do *not* contain that term -- so the comparison is drawn per row rather than
    as one shared reference line.
    """
    frame = pd.DataFrame(bias).set_index("term").sort_values("fpr_ratio")
    positions = np.arange(len(frame))

    figure, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(positions, frame["subgroup_fpr"], color=PRIMARY, label="subgroup FPR")
    ax.scatter(
        frame["background_fpr"],
        positions,
        color=ACCENT,
        marker="|",
        s=220,
        linewidths=2,
        label="background FPR (this term excluded)",
        zorder=3,
    )
    for position, (_, row) in zip(positions, frame.iterrows(), strict=True):
        ratio = row["fpr_ratio"]
        if ratio == ratio:  # skip an undefined ratio rather than printing NaN
            ax.text(
                row["subgroup_fpr"],
                position,
                f"  {ratio:.1f}x",
                va="center",
                fontsize=8,
                color="grey",
            )
    ax.set_yticks(positions, frame.index)
    ax.set(
        xlabel="false-positive rate on 'toxic'",
        title="Unintended bias by identity term",
    )
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    sns.despine(ax=ax)
    return figure


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="experiment", help="artifact prefix to read")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    sns.set_theme(style="whitegrid", context="notebook")

    config = Config(persistence_backend="file")
    config.ensure_directories()
    artifacts = get_artifact_repository(config)
    figures_dir = config.figures_dir

    def load(name: str) -> object | None:
        """Return an artifact, or None with a warning when it is absent."""
        try:
            return artifacts.load(f"{args.tag}_{name}")
        except DatasetNotFoundError:
            LOGGER.warning(
                "Artifact %s_%s is missing; skipping its figure.", args.tag, name
            )
            return None

    metadata = load("metadata")
    if metadata is None:
        LOGGER.error("Run scripts/run_experiment.py --tag %s first.", args.tag)
        return 1

    frame = load_training_data(
        config=config, repository=get_dataset_repository(config), allow_synthetic=False
    )
    # Reproduce the run's view of the data exactly, or the out-of-fold arrays
    # below would not line up with the rows they were predicted for.
    if metadata.get("sample"):
        frame = stratified_subsample(frame, metadata["sample"], config=config)
    labels = list(config.label_columns)
    y = label_matrix(frame, config)

    written: list[Path] = []
    written.append(
        save(figure_label_frequency(frame, labels), figures_dir, "label_frequency")
    )
    written.append(
        save(figure_label_cooccurrence(frame, labels), figures_dir, "label_cooccurrence")
    )
    written.append(
        save(figure_comment_length(frame, config), figures_dir, "comment_length")
    )

    leaderboard = load("leaderboard")
    if leaderboard:
        written.append(save(figure_leaderboard(leaderboard), figures_dir, "leaderboard"))
        written.append(
            save(figure_cost_vs_accuracy(leaderboard), figures_dir, "cost_vs_accuracy")
        )
        per_model = {}
        for entry in leaderboard:
            result = load(entry["model"])
            if result is not None:
                per_model[entry["model"]] = result
        if per_model:
            written.append(
                save(
                    figure_pr_auc_per_label(per_model, labels),
                    figures_dir,
                    "pr_auc_per_label",
                )
            )

    best = metadata["best_model"]
    best_result = load(best)
    oof = load("oof_scores")
    predictions = load("oof_predictions")

    if oof is not None and best_result is not None:
        tuned = np.mean(np.asarray(best_result["thresholds"], dtype=float), axis=0)
        written.append(
            save(
                figure_threshold_sweep(y, np.asarray(oof), labels, tuned),
                figures_dir,
                "threshold_sweep",
            )
        )
        # A quick consistency check: the per-label F1 recomputed here from the
        # out-of-fold scores should land near the cross-validated value.
        recomputed = per_label_metrics(y, np.asarray(oof), labels, tuned)
        stored = pd.DataFrame(best_result["label_summary"]).loc[labels, "f1"]
        drift = float((recomputed["f1"] - stored).abs().max())
        LOGGER.info("max |oof F1 - cross-validated F1| = %.4f", drift)

    if predictions is not None:
        written.append(
            save(
                figure_confusion_matrices(y, np.asarray(predictions), labels),
                figures_dir,
                "confusion_matrices",
            )
        )

    bias = load("bias")
    if bias:
        written.append(save(figure_identity_bias(bias), figures_dir, "identity_bias"))

    for path in written:
        LOGGER.info("wrote %s", path.relative_to(config.project_root))
    LOGGER.info("%d figures written to %s.", len(written), figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
