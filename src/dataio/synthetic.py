"""Synthetic stand-in for the Jigsaw dataset.

The real corpus requires Kaggle credentials and is not redistributable, so the
repository ships a generator that reproduces the *structure* of the challenge:
the same columns, the same six labels, a comparable class imbalance and the
same label hierarchy (``severe_toxic`` implies ``toxic``).

It exists so that the pipeline, the tests and the notebooks run end to end on a
clean checkout.  **No result computed on synthetic data belongs in the report**
-- every number reported must come from ``data/raw/train.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, Config

# Deliberately mild vocabulary: the point is the statistical structure, not the
# offensiveness of the strings.
_NEUTRAL = [
    "thanks for the edit, the article reads much better now",
    "i reverted the change because the source does not support it",
    "could you please add a citation for this paragraph",
    "the infobox seems out of date, i updated the release year",
    "welcome to wikipedia, feel free to use the sandbox",
    "this section duplicates the one above, merging them",
    "i disagree with the deletion but i will not edit war over it",
    "the image licence looks fine to me",
]
_INSULTS = ["you are an idiot", "what a moron", "clueless fool", "pathetic loser"]
_OBSCENE = ["this is bullshit", "what the hell is wrong with you", "damn stupid page"]
_THREATS = ["i will find you", "you are going to regret this", "watch your back"]
_IDENTITY = [
    "people like you should not edit here",
    "typical of your kind",
    "go back where you came from",
]
# Identity terms appear in *both* toxic and neutral comments on purpose, so the
# unintended-bias analysis has something real to measure.
_IDENTITY_TERMS = ["muslim", "gay", "black", "jewish", "woman", "immigrant"]
_NEUTRAL_IDENTITY = [
    "the article about the {term} community needs better sources",
    "i added a paragraph on {term} history to the page",
    "is there a category for {term} authors already",
]


def generate_synthetic_dataset(
    n_samples: int = 8_000,
    config: Config | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return a Jigsaw-shaped frame of ``n_samples`` synthetic comments.

    Args:
        n_samples: number of rows to generate.
        config: configuration providing the column names.
        random_state: seed; defaults to ``config.random_state``.

    Returns:
        A frame with ``id``, ``comment_text`` and the six label columns.
    """
    config = config or CONFIG
    rng = np.random.default_rng(
        config.random_state if random_state is None else random_state
    )

    rows: list[dict[str, object]] = []
    for index in range(n_samples):
        # ~10% toxic, matching the ratio of the real training set.
        is_toxic = rng.random() < 0.10
        labels = dict.fromkeys(config.label_columns, 0)

        if is_toxic:
            parts: list[str] = []
            labels["toxic"] = 1
            if rng.random() < 0.55:
                parts.append(rng.choice(_INSULTS))
                labels["insult"] = 1
            if rng.random() < 0.50:
                parts.append(rng.choice(_OBSCENE))
                labels["obscene"] = 1
            if rng.random() < 0.10:
                labels["severe_toxic"] = 1  # a strict subset of `toxic`
            if rng.random() < 0.04:
                parts.append(rng.choice(_THREATS))
                labels["threat"] = 1
            if rng.random() < 0.09:
                template = str(rng.choice(_IDENTITY))
                parts.append(f"{template} {rng.choice(_IDENTITY_TERMS)}")
                labels["identity_hate"] = 1
            if not parts:  # toxic but none of the finer labels fired
                parts.append(rng.choice(_INSULTS))
            text = " ".join(parts)
        elif rng.random() < 0.08:
            # Neutral comment that nevertheless mentions an identity term.
            template = str(rng.choice(_NEUTRAL_IDENTITY))
            text = template.format(term=rng.choice(_IDENTITY_TERMS))
        else:
            text = str(rng.choice(_NEUTRAL))

        rows.append(
            {
                config.id_column: f"synth{index:07d}",
                config.text_column: text,
                **labels,
            }
        )

    return pd.DataFrame(rows)
