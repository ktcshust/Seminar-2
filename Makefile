.PHONY: help setup data smoke experiment test test-all lint assets report clean

# A virtualenv puts its interpreter in bin/ on Unix and in Scripts/ on Windows.
# Detect it rather than hard-coding one, so the same target works everywhere.
VENV_BIN := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts,.venv/bin)
PYTHON   ?= $(VENV_BIN)/python
PIP      ?= $(PYTHON) -m pip

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

data:  ## Download the Jigsaw dataset from Kaggle
	$(PYTHON) scripts/download_data.py

smoke:  ## Fast end-to-end run on synthetic data (no dataset required)
	$(PYTHON) scripts/run_experiment.py --synthetic --folds 3 --backend memory --tag smoke

experiment:  ## Full 5-fold comparison on the real dataset
	$(PYTHON) scripts/run_experiment.py --folds 5 --backend file --tag experiment

test:  ## Run the test suite (fast tests only)
	$(PYTHON) -m pytest tests/ -q

test-all:  ## Run every test, including the transformer ones
	$(PYTHON) -m pytest tests/ -q -m ""

lint:  ## Check style and docstrings
	$(PYTHON) -m ruff check src tests scripts

assets:  ## Regenerate the report's tables and figures from results/
	$(PYTHON) scripts/make_report_assets.py
	$(PYTHON) scripts/make_report_figures.py

report: assets  ## Build the PDF report (regenerates the tables first)
	cd report && latexmk -pdf main.tex

clean:  ## Remove caches and build products
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
