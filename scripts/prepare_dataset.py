"""
scripts/prepare_dataset.py
===========================
Tokenizes a corpus file, creates sliding windows, and saves the processed
dataset to data/processed/tokenized.pt.

The output is a dict saved with ``torch.save``:

    {
        "input_ids" : LongTensor  shape (N, context_length),
        "target_ids": LongTensor  shape (N, context_length),
        "vocab_size": int,
        "context_length": int,
        "stride": int,
        "n_tokens": int,
        "corpus": str,          -- path of the corpus used
    }

This format is directly compatible with the existing FinanceLM training
pipeline through the ``ProcessedDataset`` class.

Default corpus
--------------
datasets/combined/corpus.txt  (MiniText8 + TinyStories)

Use --corpus to override, e.g. point to MiniText8 alone for a quick test.

Memory note
-----------
The combined corpus tokenizes to ~494M tokens.  The windowing step builds
two LongTensors of shape (N, 256) which at stride=128 means N ≈ 3.8M
windows → ~7.7 GB for input + target combined.  Ensure the machine running
this script has enough RAM (Colab Pro/A100 has 52 GB).

Usage
-----
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --corpus datasets/combined/corpus.txt
    python scripts/prepare_dataset.py --corpus data/minitext8/minitext8.txt
    python scripts/prepare_dataset.py --context-length 512 --stride 256

Arguments
---------
    --corpus          Path to training corpus (default: combined corpus).
    --context-length  Tokens per window (overrides config, default: 256).
    --stride          Step between windows (overrides config, default: 128).
    --log-level       Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from financelm.paths import (
    DATASET_CONFIG,
    PROCESSED_FILE,
    TOKENIZER_FILE,
)

_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS = _PROJECT_ROOT / "datasets" / "combined" / "corpus.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")


def _load_config() -> dict:
    try:
        import yaml  # type: ignore[import]
        with DATASET_CONFIG.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize a corpus and build a sliding-window dataset."
    )
    p.add_argument("--corpus", type=str, default=None,
                   help="Corpus file to tokenize (default: combined corpus).")
    p.add_argument("--context-length", type=int, default=None,
                   help="Tokens per window (default: from config).")
    p.add_argument("--stride", type=int, default=None,
                   help="Step between windows (default: from config).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Resolve corpus path
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.is_absolute():
            corpus_path = _PROJECT_ROOT / corpus_path
    else:
        corpus_path = _DEFAULT_CORPUS

    if not corpus_path.exists():
        logger.error("Corpus not found: %s", corpus_path)
        if corpus_path == _DEFAULT_CORPUS:
            logger.error("Run:  python scripts/build_corpus.py")
        sys.exit(1)

    if not TOKENIZER_FILE.exists():
        logger.error("Tokenizer not found: %s", TOKENIZER_FILE)
        logger.error("Run:  python scripts/train_tokenizer.py")
        sys.exit(1)

    cfg = _load_config().get("training", {})
    context_length: int = args.context_length or cfg.get("context_length", 256)
    stride: int         = args.stride         or cfg.get("stride", 128)

    corpus_mb = corpus_path.stat().st_size / (1 << 20)

    logger.info("=" * 60)
    logger.info("Script         : prepare_dataset.py")
    logger.info("Corpus         : %s (%.0f MB)", corpus_path, corpus_mb)
    logger.info("Tokenizer      : %s", TOKENIZER_FILE)
    logger.info("Context length : %d", context_length)
    logger.info("Stride         : %d", stride)
    logger.info("Output         : %s", PROCESSED_FILE)
    logger.info("=" * 60)

    # Read corpus — combined corpus is already clean, just read as-is
    logger.info("Reading corpus …")
    text = corpus_path.read_text(encoding="utf-8")
    logger.info("Read %.0f MB  |  %d chars", len(text) / (1 << 20), len(text))

    # Tokenize
    logger.info("Tokenizing …")
    tokenizer  = Tokenizer.from_file(str(TOKENIZER_FILE))
    # encode_batch is faster for large corpora but requires splitting;
    # encode handles the full string in one call — simpler and reliable.
    tokens: list[int] = tokenizer.encode(text).ids
    vocab_size = tokenizer.get_vocab_size()
    logger.info("Tokens: %d  |  Vocab: %d", len(tokens), vocab_size)

    if len(tokens) < context_length + 1:
        logger.error(
            "Corpus too small: %d tokens < context_length %d + 1",
            len(tokens), context_length,
        )
        sys.exit(1)

    # Build sliding windows
    max_start = len(tokens) - context_length - 1
    starts    = list(range(0, max_start + 1, stride))
    n_windows = len(starts)
    logger.info("Windows: %d", n_windows)

    # Allocate output tensors
    logger.info("Allocating tensors …")
    input_ids  = torch.zeros(n_windows, context_length, dtype=torch.long)
    target_ids = torch.zeros(n_windows, context_length, dtype=torch.long)

    for i, start in enumerate(tqdm(starts, desc="Windowing", unit="win")):
        end = start + context_length
        input_ids[i]  = torch.tensor(tokens[start : end],     dtype=torch.long)
        target_ids[i] = torch.tensor(tokens[start+1 : end+1], dtype=torch.long)

    payload = {
        "input_ids":      input_ids,
        "target_ids":     target_ids,
        "vocab_size":     vocab_size,
        "context_length": context_length,
        "stride":         stride,
        "n_tokens":       len(tokens),
        "corpus":         str(corpus_path),
    }

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, PROCESSED_FILE)

    file_mb = PROCESSED_FILE.stat().st_size / (1 << 20)
    logger.info("Saved %d windows → %s (%.1f MB)", n_windows, PROCESSED_FILE, file_mb)
    logger.info("=" * 60)
    logger.info("Done.  Run verify_dataset.py to validate the output.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
