"""
scripts/train_tokenizer.py
===========================
Trains a BPE tokenizer on the MiniText8 corpus and saves it to
data/tokenizer/tokenizer.json.

Reuses the existing HuggingFace ``tokenizers`` BPE implementation.
Vocabulary size and special tokens are read from configs/dataset.yaml.

Usage
-----
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 8192

Arguments
---------
    --vocab-size    BPE vocabulary size (overrides config, default: 16384).
    --log-level     Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from financelm.paths import (
    DATASET_CONFIG,
    MINITEXT8_FILE,
    TOKENIZER_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_tokenizer")


def _load_config() -> dict:
    """Load dataset.yaml; return empty dict if PyYAML is unavailable."""
    try:
        import yaml  # type: ignore[import]
        with DATASET_CONFIG.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BPE tokenizer on MiniText8.")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Vocabulary size (default: from configs/dataset.yaml).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not MINITEXT8_FILE.exists():
        logger.error("Corpus not found: %s", MINITEXT8_FILE)
        logger.error("Run:  python scripts/download_minitext8.py")
        sys.exit(1)

    cfg = _load_config()
    vocab_size: int = (
        args.vocab_size
        or cfg.get("tokenizer", {}).get("vocab_size", 16384)
    )
    special_tokens: list[str] = cfg.get("tokenizer", {}).get(
        "special_tokens", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    logger.info("=" * 60)
    logger.info("Script       : train_tokenizer.py")
    logger.info("Corpus       : %s", MINITEXT8_FILE)
    logger.info("Vocab size   : %d", vocab_size)
    logger.info("Special toks : %s", special_tokens)
    logger.info("Output       : %s", TOKENIZER_FILE)
    logger.info("=" * 60)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    logger.info("Training BPE tokenizer …")
    tokenizer.train(files=[str(MINITEXT8_FILE)], trainer=trainer)

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_FILE))

    logger.info("Tokenizer saved → %s", TOKENIZER_FILE)
    logger.info("Vocabulary size : %d", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
