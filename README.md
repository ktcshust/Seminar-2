# Toxic Comment Classification

Seminar project, topic 3: multi-label classification of Wikipedia talk-page
comments from the
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
into six non-exclusive categories: `toxic`, `severe_toxic`, `obscene`, `threat`,
`insult`, `identity_hate`.

The project compares two trivial baselines, three linear models over TF-IDF
features and a fine-tuned DistilBERT under 5-fold multi-label stratified cross
validation, then analyses where the best model fails and whether it is biased
against identity terms.

## Results

5-fold cross validation on the full training set (159,571 comments), iterative
stratification, seed 42.  Thresholds are tuned per label on a hold-out carved
from each training fold, never on the validation fold.

| Model | Mean column ROC-AUC | PR-AUC | Macro F1 | Micro F1 | Fit time / fold |
|---|---|---|---|---|---|
| **distilbert** (fine-tuned, RTX 4050 GPU) | **0.9898 ± 0.0011** | **0.7193** | **0.6827** | **0.7870** | 2238 s |
| logreg (TF-IDF word + char, logistic regression) | 0.9841 ± 0.0010 | 0.6726 | 0.6410 | 0.7551 | 232 s |
| nbsvm (NB log-count ratio + logistic regression) | 0.9830 ± 0.0010 | 0.6665 | 0.6350 | 0.7566 | 238 s |
| linear_svm | 0.9800 ± 0.0008 | 0.6559 | 0.6306 | 0.7486 | 204 s |
| keyword (profanity list, baseline) | 0.7831 ± 0.0075 | 0.1298 | 0.2485 | 0.2821 | 0.6 s |
| prior (label frequency, baseline) | 0.5000 | 0.0367 | 0.0616 | 0.1231 | 0.0 s |

Mean column-wise ROC-AUC is the competition's official metric; the winning
private-leaderboard score was about 0.988.  The full tables, per-label metrics,
significance tests and figures are in [`report/`](report/).

The linear models and DistilBERT were run separately (CPU and GPU) on the same
folds and merged with `scripts/merge_runs.py`; DistilBERT wins on all five
folds.

Key findings:

* Every learned model beats the keyword baseline by 0.20 AUC, yet the rare
  labels stay hard: even DistilBERT reaches a PR-AUC of only 0.50 on
  `severe_toxic`, 0.54 on `threat` (96 positives per fold) and 0.61 on
  `identity_hate`.
* DistilBERT gains most where context matters: the rare labels, profanity
  that is quoted rather than aimed at someone, and the label hierarchy, which
  it learns on its own (it almost never predicts a sub-label without `toxic`).
* Per-label threshold tuning is essential for DistilBERT (+0.17 F1 on
  `severe_toxic`, +0.11 on `threat`, thresholds 0.25-0.49) but nearly useless
  for the linear models, whose balanced class weights already shift their
  scores.
* Unintended bias: a clean comment that mentions *gay* is flagged as toxic
  6.8 times as often as the average clean comment by DistilBERT, and 10.5
  times by logistic regression.

## Repository layout

```
src/
  config.py            single switch for the persistence backend
  dataio/              loading, schema validation, synthetic stand-in data
  preprocessing/       text cleaner and TF-IDF vectorisers
  models/              baselines, linear models, DistilBERT, model registry
  evaluation/          multi-label folds, metrics, thresholds, statistics, bias
  repository/          file-system and in-memory persistence behind one interface
scripts/
  download_data.py     fetch the Kaggle data
  run_experiment.py    cross-validate the selected models, store every artifact
  merge_runs.py        fold a second run (e.g. the GPU one) into the main one
  make_report_assets.py / make_report_figures.py   tables and figures for the report
notebooks/             exploratory analysis, model comparison, error and bias analysis
results/               stored experiment artifacts (JSON + joblib)
report/                LaTeX report, generated tables and figures
tests/                 pytest suite
```

## Running it

### Locally

Python 3.10 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements-dev.txt
```

Download the data.  This needs a Kaggle API token in `~/.kaggle/access_token`
(or a classic `~/.kaggle/kaggle.json`), and the competition rules must be
accepted on the Kaggle website first:

```bash
python scripts/download_data.py
```

Run the experiment and build the report:

```bash
python scripts/run_experiment.py --folds 5 --tag experiment   # about 1 h on a laptop CPU
python scripts/make_report_assets.py
python scripts/make_report_figures.py
cd report && latexmk -pdf main.tex
```

With `make` available, the same steps are `make setup data experiment report`.
`make smoke` runs the whole pipeline in seconds on synthetic data, without the
download; its numbers prove the wiring only and must not be reported.

### DistilBERT

The transformer needs PyTorch with CUDA.  Install it into the environment, run
it as a separate experiment with the same seed and folds, and merge it into the
main run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dl.txt
python scripts/run_experiment.py --models distilbert --folds 5 --tag dl
python scripts/merge_runs.py --base experiment --extra dl
```

### Google Colab

Open any notebook in Colab.  Its first cell clones this repository, installs the
dependencies and downloads the data; store your Kaggle token under Colab
*Secrets* as `KAGGLE_API_TOKEN` beforehand.  For DistilBERT, select a GPU
runtime.

### Notebooks

| Notebook | Content |
|---|---|
| `01_exploratory_analysis` | label frequency, co-occurrence, hierarchy, comment length, cleaning |
| `02_model_comparison` | folds, cross validation, leaderboard, per-label results, significance |
| `03_error_and_bias_analysis` | threshold sweep, confusion matrices, error examples, identity bias |

Notebooks 02 and 03 cross-validate on a stratified 20,000-comment subsample so
they run in minutes; the numbers in the report come from the full run in
`results/`.

## Persistence

All data and results go through a repository interface with two
implementations, file system and in memory.  Switching between them is one
value: `persistence_backend` in `src/config.py`, the `TCC_PERSISTENCE`
environment variable, or `--backend file|memory` on the command line.

## Tests

```bash
python -m pytest tests/          # fast suite
python -m pytest tests/ -m ""    # include the transformer tests (needs PyTorch)
```
