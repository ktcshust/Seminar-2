#!/usr/bin/env python3
"""Fetch the Jigsaw dataset, or generate the synthetic stand-in.

The competition data cannot be redistributed, so this script only automates the
Kaggle download and verifies the result.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG  # noqa: E402
from src.dataio import generate_synthetic_dataset, raw_files_available  # noqa: E402

COMPETITION = "jigsaw-toxic-comment-classification-challenge"
EXPECTED = ("train.csv", "test.csv", "test_labels.csv")


def download_from_kaggle(raw_dir: Path) -> bool:
    """Download and unzip the competition files.  Returns success."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _download_with_kagglehub(raw_dir) or _download_with_cli(raw_dir):
        _extract_archives(raw_dir)
        return True
    print(
        "Kaggle download failed.  Install the clients with "
        "`pip install kagglehub kaggle` into this environment, then "
        "authenticate one of two ways:\n"
        "  - an API token from kaggle.com/settings/api saved to "
        "~/.kaggle/access_token (or exported as KAGGLE_API_TOKEN).  "
        "Tokens issued today look like `KGAT_...` and only kagglehub "
        "understands them; the `kaggle` CLI still wants the older "
        "kaggle.json.\n"
        "  - a kaggle.json username/key pair in ~/.kaggle/.\n"
        "The competition rules must also be accepted on the website, "
        "otherwise the API answers 403 even with valid credentials.",
        file=sys.stderr,
    )
    return False


def _download_with_kagglehub(raw_dir: Path) -> bool:
    """Try the modern client.  It is the only one that accepts KGAT_ tokens."""
    try:
        import kagglehub
    except ImportError:
        return False

    # kagglehub reads KAGGLE_API_TOKEN from the environment but not the
    # ~/.kaggle/access_token file that its own docs tell you to write.
    token_file = Path.home() / ".kaggle" / "access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()

    try:
        cached = Path(kagglehub.competition_download(COMPETITION))
    except Exception as exc:  # noqa: BLE001 - any failure means "try the CLI"
        print(f"kagglehub download failed: {exc}", file=sys.stderr)
        return False

    # kagglehub unpacks into its own cache; copy the archives across so that
    # raw_dir owns them and _extract_archives can do the nested unzip.
    for source in cached.iterdir():
        if source.is_file():
            shutil.copy2(source, raw_dir / source.name)
    return True


def _download_with_cli(raw_dir: Path) -> bool:
    """Fall back to the classic CLI for kaggle.json credentials."""
    # `python -m kaggle.cli` rather than a bare `kaggle`: the CLI lives in
    # whichever environment this script is running under, which is not
    # necessarily on PATH, and the `kaggle` package itself has no __main__.
    command = [
        sys.executable,
        "-m",
        "kaggle.cli",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(raw_dir),
    ]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"kaggle CLI download failed: {exc}", file=sys.stderr)
        return False
    return True


def _extract_archives(raw_dir: Path) -> None:
    """Unzip every archive in ``raw_dir``, including nested ones.

    Kaggle ships this competition as a zip of zips: the outer archive expands to
    ``train.csv.zip``, ``test.csv.zip`` and friends, so a single pass leaves no
    CSV behind at all.
    """
    extracted: set[Path] = set()
    while True:
        pending = [path for path in raw_dir.glob("*.zip") if path not in extracted]
        if not pending:
            return
        for archive in pending:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(raw_dir)
            extracted.add(archive)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="write a synthetic train.csv instead of downloading",
    )
    parser.add_argument("--n-samples", type=int, default=8000)
    args = parser.parse_args()

    raw_dir = CONFIG.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        target = raw_dir / "train_synthetic.csv"
        generate_synthetic_dataset(args.n_samples).to_csv(target, index=False)
        print(f"Wrote {args.n_samples} SYNTHETIC rows to {target}")
        print("These are for smoke-testing the pipeline only - never for the report.")
        return 0

    if raw_files_available():
        print(f"Dataset already present in {raw_dir}")
        return 0

    # A previous run may have downloaded the archives but failed to expand them.
    if any(raw_dir.glob("*.zip")):
        print(f"Expanding archives already present in {raw_dir}")
        _extract_archives(raw_dir)
        if raw_files_available():
            print(f"Dataset ready in {raw_dir}")
            return 0

    if not download_from_kaggle(raw_dir):
        return 1

    missing = [name for name in EXPECTED if not (raw_dir / name).exists()]
    if missing:
        print(
            f"Download finished but these files are missing: {missing}",
            file=sys.stderr,
        )
        return 1
    print(f"Dataset ready in {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
