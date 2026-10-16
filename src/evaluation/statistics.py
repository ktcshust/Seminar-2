"""Statistical comparison of two models.

"Model A scored 0.978 and model B 0.976" is not a result: with five folds the
difference may well be noise.  Every claim of superiority in the report is
backed by a paired test over the *same* folds, which is why
:func:`~src.evaluation.cross_validation.cross_validate_model` accepts a
pre-computed fold assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PairedComparison:
    """Outcome of comparing two models over the same folds."""

    model_a: str
    model_b: str
    mean_a: float
    mean_b: float
    mean_difference: float
    cohens_d: float
    wilcoxon_p: float
    ttest_p: float
    n_folds: int

    @property
    def significant(self, alpha: float = 0.05) -> bool:
        """Whether the Wilcoxon test rejects at the 5% level."""
        return bool(self.wilcoxon_p < alpha)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view."""
        return {**self.__dict__, "significant": self.significant}


def paired_comparison(
    scores_a: np.ndarray, scores_b: np.ndarray, model_a: str, model_b: str
) -> PairedComparison:
    """Compare two models from their per-fold scores.

    Args:
        scores_a: per-fold metric values of the first model.
        scores_b: per-fold metric values of the second model, same folds and
            same order.
        model_a: name of the first model.
        model_b: name of the second model.

    Returns:
        A :class:`PairedComparison`.  With the usual five folds the Wilcoxon
        test cannot produce a p-value below 0.0625 whatever the data, so treat
        it as weak evidence and report the effect size alongside it.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if scores_a.shape != scores_b.shape:
        raise ValueError("Both models must be scored on the same number of folds.")

    differences = scores_a - scores_b
    spread = differences.std(ddof=1)

    if np.allclose(differences, 0):
        wilcoxon_p = ttest_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(scores_a, scores_b).pvalue)
        ttest_p = float(stats.ttest_rel(scores_a, scores_b).pvalue)

    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        mean_a=float(scores_a.mean()),
        mean_b=float(scores_b.mean()),
        mean_difference=float(differences.mean()),
        cohens_d=float(differences.mean() / spread) if spread > 0 else 0.0,
        wilcoxon_p=wilcoxon_p,
        ttest_p=ttest_p,
        n_folds=int(len(scores_a)),
    )


def comparison_matrix(
    fold_scores: dict[str, np.ndarray], alpha: float = 0.05
) -> pd.DataFrame:
    """Return the pairwise p-value matrix for every model in ``fold_scores``.

    Note that running all pairs inflates the family-wise error rate; with ``m``
    comparisons, apply a Bonferroni correction by testing against
    ``alpha / m`` -- the corrected level is returned in the frame's ``attrs``.
    """
    names = list(fold_scores)
    matrix = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for first in names:
        for second in names:
            if first == second:
                matrix.loc[first, second] = 1.0
            else:
                matrix.loc[first, second] = paired_comparison(
                    fold_scores[first], fold_scores[second], first, second
                ).wilcoxon_p
    n_pairs = max(1, len(names) * (len(names) - 1) // 2)
    matrix.attrs["bonferroni_alpha"] = alpha / n_pairs
    return matrix
