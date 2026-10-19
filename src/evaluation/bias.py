"""Unintended-bias analysis.

Models trained on this corpus learn that comments mentioning an identity term
are frequently toxic, and then flag *any* comment containing the term -- "I am a
gay man" gets a high toxicity score.  Jigsaw ran an entire follow-up competition
on the problem, and quantifying it turns a table of AUC values into an actual
finding worth discussing.

The measurement is a subgroup comparison: for each identity term, compare the
false-positive rate among comments containing it with the rate everywhere else.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import _safe_roc_auc

#: Terms that appear frequently in the corpus and denote a protected group.
DEFAULT_IDENTITY_TERMS: tuple[str, ...] = (
    "gay",
    "lesbian",
    "transgender",
    "muslim",
    "christian",
    "jewish",
    "black",
    "white",
    "woman",
    "female",
    "immigrant",
)


def _false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return FP / (FP + TN), or NaN when there is no true negative."""
    negatives = y_true == 0
    if not negatives.any():
        return float("nan")
    return float(y_pred[negatives].mean())


def identity_term_report(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_index: int = 0,
    terms: Sequence[str] = DEFAULT_IDENTITY_TERMS,
    threshold: float = 0.5,
    min_occurrences: int = 20,
) -> pd.DataFrame:
    """Compare model behaviour on comments mentioning each identity term.

    Args:
        texts: raw comments, aligned with the label matrix.
        y_true: binary label matrix ``(n_samples, n_labels)``.
        y_score: predicted scores, same shape.
        label_index: which label to audit; 0 is ``toxic``.
        terms: identity terms to test.
        threshold: decision cut-off used to form the predictions.
        min_occurrences: skip terms with fewer occurrences than this, where the
            estimated rates would be too noisy to interpret.

    Returns:
        One row per term with the subgroup size, its positive rate, the
        subgroup and background false-positive rates, their ratio and the
        subgroup AUC.  A ratio well above 1 is the signature of unintended bias:
        the model penalises the term itself rather than the message.
    """
    lowered = [str(text).lower() for text in texts]
    truth = np.asarray(y_true)[:, label_index]
    predictions = (np.asarray(y_score)[:, label_index] >= threshold).astype(int)
    scores = np.asarray(y_score)[:, label_index]

    rows = []
    for term in terms:
        mask = np.array([term in text for text in lowered], dtype=bool)
        if mask.sum() < min_occurrences:
            continue
        subgroup_fpr = _false_positive_rate(truth[mask], predictions[mask])
        background_fpr = _false_positive_rate(truth[~mask], predictions[~mask])
        rows.append(
            {
                "term": term,
                "n_comments": int(mask.sum()),
                "positive_rate": float(truth[mask].mean()),
                "subgroup_fpr": subgroup_fpr,
                "background_fpr": background_fpr,
                "fpr_ratio": (
                    subgroup_fpr / background_fpr if background_fpr else float("nan")
                ),
                "subgroup_auc": _safe_roc_auc(truth[mask], scores[mask]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "term",
                "n_comments",
                "positive_rate",
                "subgroup_fpr",
                "background_fpr",
                "fpr_ratio",
                "subgroup_auc",
            ]
        ).set_index("term")

    return pd.DataFrame(rows).set_index("term").sort_values("fpr_ratio", ascending=False)


def error_examples(
    texts: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_index: int = 0,
    threshold: float = 0.5,
    n_examples: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return the most confident false positives and false negatives.

    Reading these is the fastest route to an error taxonomy for the report:
    sarcasm, quoted insults, non-English comments, and so on.
    """
    truth = np.asarray(y_true)[:, label_index]
    scores = np.asarray(y_score)[:, label_index]
    frame = pd.DataFrame({"text": list(texts), "y_true": truth, "score": scores})

    false_positives = frame[(frame.y_true == 0) & (frame.score >= threshold)]
    false_negatives = frame[(frame.y_true == 1) & (frame.score < threshold)]
    return {
        "false_positives": false_positives.nlargest(n_examples, "score"),
        "false_negatives": false_negatives.nsmallest(n_examples, "score"),
    }
