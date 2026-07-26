"""
scripts/download_tinystories.py
================================
Downloads the official TinyStories dataset from Hugging Face and saves
every story as plain text under datasets/tinystories/.

Dataset
-------
    roneneldan/TinyStories  (Hugging Face)
    https://huggingface.co/datasets/roneneldan/TinyStories

    Splits: train, validation
    Column: text  (one story per row)

Output
------
    datasets/tinystories/train.txt      — one story per line, blank line between
    datasets/tinystories/validation.txt
    datasets/tinystories/stats.json     — row counts, character counts, file sizes

Usage
-----
    python scripts/download_tinystories.py
    python scripts/download_tinystories.py --split train
    python scripts/download_tinystories.py --force

Arguments
---------
    --split     Which split to download: train | validation | all (default: all).
    --force     Re-download even if files already exist.
    --log-level Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_tinystories")

DATASET_NAME = "roneneldan/TinyStories"
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "tinystories"
SPLITS       = ["train", "validation"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the TinyStories dataset.")
    p.add_argument("--split",      default="all", choices=["train", "validation", "all"])
    p.add_argument("--force",      action="store_true",
                   help="Re-download even if files already exist.")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _download_split(split: str, force: bool) -> dict:
    """
    Download one split and write it to ``datasets/tinystories/{split}.txt``.

    Returns a stats dict: rows, characters, file_size_bytes.
    """
    out_path = DATASETS_DIR / f"{split}.txt"

    if out_path.exists() and not force:
        logger.info("Already exists — skipping %s (use --force to re-download)", out_path)
        size = out_path.stat().st_size
        # Count lines (= stories) without reading everything into RAM
        rows = sum(1 for line in out_path.open(encoding="utf-8") if line.strip())
        return {"split": split, "rows": rows, "file_size_bytes": size, "skipped": True}

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        logger.error("The 'datasets' package is required: pip install datasets")
        sys.exit(1)

    logger.info("Downloading split='%s' from %s …", split, DATASET_NAME)
    t0 = time.perf_counter()

    ds = load_dataset(DATASET_NAME, split=split)

    elapsed_dl = time.perf_counter() - t0
    logger.info("Downloaded %d rows in %.1fs", len(ds), elapsed_dl)

    # Write stories to plain text, separated by a blank line
    logger.info("Writing → %s", out_path)
    t1 = time.perf_counter()
    total_chars = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            text = row["text"].strip()
            if not text:
                continue
            fh.write(text)
            fh.write("\n\n")          # blank line between stories
            total_chars += len(text)

    elapsed_write = time.perf_counter() - t1
    file_size     = out_path.stat().st_size

    logger.info(
        "Saved %d stories  |  %d chars  |  %.1f MB  |  %.1fs",
        len(ds), total_chars, file_size / (1 << 20), elapsed_write,
    )

    return {
        "split":           split,
        "rows":            len(ds),
        "characters":      total_chars,
        "file_size_bytes": file_size,
        "skipped":         False,
    }


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    splits_to_download = SPLITS if args.split == "all" else [args.split]

    logger.info("=" * 60)
    logger.info("Script  : download_tinystories.py")
    logger.info("Dataset : %s", DATASET_NAME)
    logger.info("Splits  : %s", splits_to_download)
    logger.info("Output  : %s", DATASETS_DIR)
    logger.info("=" * 60)

    all_stats: list[dict] = []
    try:
        for split in splits_to_download:
            stats = _download_split(split, force=args.force)
            all_stats.append(stats)
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Download failed: %s", exc, exc_info=True)
        sys.exit(1)

    # Save stats JSON
    stats_path = DATASETS_DIR / "stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2)
    logger.info("Stats saved → %s", stats_path)

    # Summary
    logger.info("=" * 60)
    for s in all_stats:
        skipped = " (skipped — already exists)" if s.get("skipped") else ""
        rows    = f"{s['rows']:,}"
        mb      = f"{s['file_size_bytes'] / (1<<20):.1f} MB"
        logger.info("%-12s  rows=%-10s  size=%s%s", s["split"], rows, mb, skipped)
    logger.info("Done.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
