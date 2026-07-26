"""
scripts/train_tokenizer.py
===========================
Trains a BPE tokenizer on a corpus file and saves it to
data/tokenizer/tokenizer.json.

Reuses the existing HuggingFace ``tokenizers`` BPE implementation.
Vocabulary size and special tokens are read from configs/dataset.yaml.

By default trains on the combined corpus (MiniText8 + TinyStories) at
datasets/combined/corpus.txt.  Pass --corpus to override.

Usage
-----
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --corpus datasets/combined/corpus.txt
    python scripts/train_tokenizer.py --vocab-size 16384

Arguments
---------
    --corpus        Path to training corpus (default: combined corpus).
    --vocab-size    BPE vocabulary size (overrides config).
    --log-level     Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from financelm.paths import (
    DATASET_CONFIG,
    TOKENIZER_FILE,
)

# Default corpus: combined MiniText8 + TinyStories
_PROJECT_ROOT    = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS  = _PROJECT_ROOT / "datasets" / "combined" / "corpus.txt"

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
    p = argparse.ArgumentParser(description="Train BPE tokenizer.")
    p.add_argument("--corpus",     type=str, default=None,
                   help="Path to training corpus (default: datasets/combined/corpus.txt).")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Vocabulary size (default: from configs/dataset.yaml).")
    p.add_argument("--log-level",  default="INFO",
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

    cfg = _load_config()
    vocab_size: int = (
        args.vocab_size
        or cfg.get("tokenizer", {}).get("vocab_size", 8192)
    )
    special_tokens: list[str] = cfg.get("tokenizer", {}).get(
        "special_tokens", ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    corpus_mb = corpus_path.stat().st_size / (1 << 20)

    logger.info("=" * 60)
    logger.info("Script       : train_tokenizer.py")
    logger.info("Corpus       : %s (%.0f MB)", corpus_path, corpus_mb)
    logger.info("Vocab size   : %d", vocab_size)
    logger.info("Special toks : %s", special_tokens)
    logger.info("Output       : %s", TOKENIZER_FILE)
    logger.info("=" * 60)

    # ByteLevel pre-tokenizer: handles mixed case, punctuation, unicode.
    # The combined corpus includes TinyStories (capitalised, punctuated text)
    # so Whitespace pre-tokenization would miss subwords across capitalisation.
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    logger.info("Training BPE tokenizer …")
    t0 = time.perf_counter()
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1fs", elapsed)

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_FILE))

    logger.info("Tokenizer saved → %s", TOKENIZER_FILE)
    logger.info("Vocabulary size : %d", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
