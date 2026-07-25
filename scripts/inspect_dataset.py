"""
scripts/inspect_dataset.py
===========================
Displays lightweight statistics about the MiniText8 corpus without loading
the whole file into memory more than once.

Usage
-----
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --sample-chars 500

Arguments
---------
    --sample-chars  Characters of sample text to display (default: 300).
    --log-level     Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from financelm.paths import MINITEXT8_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inspect_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect the MiniText8 corpus.")
    p.add_argument("--sample-chars", type=int, default=300,
                   help="Characters of sample text to display (default: 300).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not MINITEXT8_FILE.exists():
        logger.error("MiniText8 not found: %s", MINITEXT8_FILE)
        logger.error("Run:  python scripts/download_minitext8.py")
        sys.exit(1)

    logger.info("Reading corpus …")
    text = MINITEXT8_FILE.read_text(encoding="utf-8")

    # --- basic counts ---
    n_chars = len(text)
    words   = text.split()
    n_words = len(words)

    # MiniText8 is a single continuous string (no newlines in text8),
    # so we split into artificial "samples" of 1000 words for statistics.
    chunk = 1000
    samples = [" ".join(words[i : i + chunk]) for i in range(0, n_words, chunk)]
    n_samples = len(samples)

    lengths = [len(s) for s in samples]
    avg_len = sum(lengths) / n_samples if n_samples else 0
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    # Vocabulary (unique whitespace tokens — not BPE)
    vocab = set(words)
    n_vocab = len(vocab)

    sep = "=" * 56
    print(sep)
    print("  MiniText8 Dataset Inspection")
    print(sep)
    print(f"  File           : {MINITEXT8_FILE}")
    print(f"  Size           : {MINITEXT8_FILE.stat().st_size:,} bytes")
    print(f"  Characters     : {n_chars:,}")
    print(f"  Words          : {n_words:,}")
    print(f"  Samples (1k-w) : {n_samples:,}")
    print(f"  Vocab (words)  : {n_vocab:,}")
    print()
    print("  Sample statistics (chars per 1k-word chunk):")
    print(f"    Average : {avg_len:,.0f}")
    print(f"    Longest : {max_len:,}")
    print(f"    Shortest: {min_len:,}")
    print()
    print("  Sample text (first characters):")
    print(f"    {text[:args.sample_chars]!r}")
    print(sep)


if __name__ == "__main__":
    main()
