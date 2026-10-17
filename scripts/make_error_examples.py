#!/usr/bin/env python3
"""Surface the misclassifications behind the error taxonomy in the report.

The taxonomy in Section 7 needs concrete examples, not adjectives.  This script
reads the stored out-of-fold predictions of the best model and writes, for each
label, the false positives and false negatives the model was *most confident*
about -- those are the informative ones, because a near-threshold mistake says
nothing except that the threshold had to fall somewhere.

Categorising what comes out is a human job; this only finds the candidates.

    python scripts/make_error_examples.py --label toxic --n 15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG  # noqa: E402
from src.dataio import label_matrix, load_training_data, texts  # noqa: E402

LOGGER = logging.getLogger("error_examples")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="experiment", help="artifact prefix to read")
    parser.add_argument(
        "--label",
        default=None,
        help="restrict to one label (default: every label)",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="examples per error type per label"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="truncate each comment to this many characters",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the candidates to this JSON file",
    )
    return parser.parse_args(argv)


def _mean_thresholds(result_path: Path) -> np.ndarray:
    """Average the per-fold tuned thresholds into one vector per label.

    The out-of-fold scores come from five different fitted models, each with its
    own tuned cut-off.  Averaging is the honest summary for a post-hoc read of
    the pooled predictions; it is not what the reported metrics used.
    """
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    return np.asarray(stored["thresholds"], dtype=float).mean(axis=0)


def collect(args: argparse.Namespace) -> dict[str, dict[str, list[dict]]]:
    """Return the most confident errors per label, keyed by error type."""
    results_dir = CONFIG.results_dir
    scores = joblib.load(results_dir / f"{args.tag}_oof_scores.joblib")
    # The stored out-of-fold scores belong to whichever model won the run, so
    # the thresholds have to come from that same model -- naming one here would
    # silently mismatch the scores the next time the ranking changes.
    best = json.loads(
        (results_dir / f"{args.tag}_metadata.json").read_text(encoding="utf-8")
    )["best_model"]
    LOGGER.info("reading out-of-fold predictions of %s", best)
    thresholds = _mean_thresholds(results_dir / f"{args.tag}_{best}.json")

    frame = load_training_data(config=CONFIG)
    comment_text = texts(frame, CONFIG).tolist()
    y = label_matrix(frame, CONFIG)

    if len(comment_text) != scores.shape[0]:
        raise SystemExit(
            f"{len(comment_text)} comments but {scores.shape[0]} stored predictions; "
            "the run and the corpus do not match."
        )

    labels = list(CONFIG.label_columns)
    wanted = [args.label] if args.label else labels

    found: dict[str, dict[str, list[dict]]] = {}
    for name in wanted:
        index = labels.index(name)
        truth = y[:, index].astype(bool)
        score = scores[:, index]
        predicted = score >= thresholds[index]

        # Most confident first: a mistake made at 0.99 is a different animal
        # from one made at 0.51.
        false_positive = np.flatnonzero(predicted & ~truth)
        false_negative = np.flatnonzero(~predicted & truth)
        false_positive = false_positive[np.argsort(-score[false_positive])]
        false_negative = false_negative[np.argsort(score[false_negative])]

        found[name] = {
            "false_positive": _rows(
                false_positive[: args.n], comment_text, score, args.max_chars
            ),
            "false_negative": _rows(
                false_negative[: args.n], comment_text, score, args.max_chars
            ),
        }
        LOGGER.info(
            "%s: %d false positives, %d false negatives (threshold %.3f)",
            name,
            false_positive.size,
            false_negative.size,
            thresholds[index],
        )
    return found


def _rows(
    indices: np.ndarray, comment_text: list[str], score: np.ndarray, max_chars: int
) -> list[dict]:
    """Package the selected comments with their scores."""
    return [
        {
            "row": int(i),
            "score": round(float(score[i]), 4),
            "text": " ".join(comment_text[i].split())[:max_chars],
        }
        for i in indices
    ]


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    found = collect(args)

    for name, groups in found.items():
        for kind, rows in groups.items():
            print("\n" + "=" * 78)
            print(f"{name.upper()}  --  {kind.replace('_', ' ')}  (most confident)")
            print("=" * 78)
            for row in rows:
                print(f"\n[row {row['row']}  score {row['score']:.4f}]")
                print(textwrap.fill(row["text"], width=76))

    if args.out:
        args.out.write_text(json.dumps(found, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
