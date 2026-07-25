"""
scripts/prepare_dataset.py
===========================
Tokenizes the MiniText8 corpus, creates sliding windows, and saves the
processed dataset to data/processed/tokenized.pt.

The output is a dict saved with ``torch.save``:

    {
        "input_ids" : LongTensor  shape (N, context_length),
        "target_ids": LongTensor  shape (N, context_length),
        "vocab_size": int,
        "context_length": int,
        "stride": int,
        "n_tokens": int,
    }

This format is directly compatible with the existing FinanceLM training
pipeline through the ``TokenizerDataset`` class.

Usage
-----
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --context-length 512 --stride 256

Arguments
---------
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

from financelm.data.loader import load_minitext8
from financelm.data.preprocessing import preprocess
from financelm.paths import (
    DATASET_CONFIG,
    MINITEXT8_FILE,
    PROCESSED_FILE,
    TOKENIZER_FILE,
)

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
        description="Tokenize MiniText8 and build sliding-window dataset."
    )
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

    for path, hint in [
        (MINITEXT8_FILE, "python scripts/download_minitext8.py"),
        (TOKENIZER_FILE, "python scripts/train_tokenizer.py"),
    ]:
        if not path.exists():
            logger.error("Not found: %s", path)
            logger.error("Run:  %s", hint)
            sys.exit(1)

    cfg = _load_config().get("training", {})
    context_length: int = args.context_length or cfg.get("context_length", 256)
    stride: int         = args.stride         or cfg.get("stride", 128)

    logger.info("=" * 60)
    logger.info("Script         : prepare_dataset.py")
    logger.info("Corpus         : %s", MINITEXT8_FILE)
    logger.info("Tokenizer      : %s", TOKENIZER_FILE)
    logger.info("Context length : %d", context_length)
    logger.info("Stride         : %d", stride)
    logger.info("Output         : %s", PROCESSED_FILE)
    logger.info("=" * 60)

    # Load and preprocess
    logger.info("Loading and preprocessing corpus …")
    text = preprocess(load_minitext8(MINITEXT8_FILE))

    # Tokenize
    logger.info("Tokenizing …")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
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
    starts = list(range(0, max_start + 1, stride))
    n_windows = len(starts)
    logger.info("Windows: %d", n_windows)

    input_ids  = torch.zeros(n_windows, context_length, dtype=torch.long)
    target_ids = torch.zeros(n_windows, context_length, dtype=torch.long)

    for i, start in enumerate(tqdm(starts, desc="Windowing", unit="win")):
        end = start + context_length
        input_ids[i]  = torch.tensor(tokens[start:end],      dtype=torch.long)
        target_ids[i] = torch.tensor(tokens[start+1:end+1],  dtype=torch.long)

    payload = {
        "input_ids":      input_ids,
        "target_ids":     target_ids,
        "vocab_size":     vocab_size,
        "context_length": context_length,
        "stride":         stride,
        "n_tokens":       len(tokens),
    }

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, PROCESSED_FILE)

    file_mb = PROCESSED_FILE.stat().st_size / (1 << 20)
    logger.info("Saved %d windows → %s (%.1f MB)", n_windows, PROCESSED_FILE, file_mb)


if __name__ == "__main__":
    main()
