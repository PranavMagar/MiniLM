"""
scripts/verify_dataset.py
==========================
Verifies that all pipeline artefacts exist and are internally consistent.

Checks performed
----------------
1. MiniText8 corpus exists and is non-empty.
2. Tokenizer exists and loads without error.
3. Processed dataset exists.
4. input_ids and target_ids shapes are equal.
5. Sequence lengths match the configured context_length.
6. No empty (all-zero) sequences.
7. All token IDs are within vocabulary range [0, vocab_size).

Prints a concise PASS / FAIL report and exits with code 0 on success,
1 on failure.

Usage
-----
    python scripts/verify_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tokenizers import Tokenizer

from financelm.paths import MINITEXT8_FILE, PROCESSED_FILE, TOKENIZER_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify_dataset")


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

Check = tuple[str, bool, str]   # (name, passed, detail)


def _check(name: str, fn: Callable[[], str]) -> Check:
    """Run *fn*; return a Check tuple.  *fn* returns a detail string on pass
    or raises an exception on failure."""
    try:
        detail = fn()
        return name, True, detail
    except Exception as exc:
        return name, False, str(exc)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _corpus_exists() -> str:
    if not MINITEXT8_FILE.exists():
        raise FileNotFoundError(f"Not found: {MINITEXT8_FILE}")
    size = MINITEXT8_FILE.stat().st_size
    if size == 0:
        raise ValueError("File is empty.")
    return f"{size:,} bytes"


def _tokenizer_exists() -> str:
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(f"Not found: {TOKENIZER_FILE}")
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    return f"vocab_size={tok.get_vocab_size()}"


def _processed_exists() -> str:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Not found: {PROCESSED_FILE}")
    return f"{PROCESSED_FILE.stat().st_size:,} bytes"


def _shapes_match() -> str:
    data = torch.load(PROCESSED_FILE, map_location="cpu", weights_only=True)
    inp, tgt = data["input_ids"], data["target_ids"]
    if inp.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: input={inp.shape} target={tgt.shape}")
    return f"shape={tuple(inp.shape)}"


def _seq_lengths() -> str:
    data = torch.load(PROCESSED_FILE, map_location="cpu", weights_only=True)
    stored_cl = data["context_length"]
    actual_cl = data["input_ids"].shape[1]
    if stored_cl != actual_cl:
        raise ValueError(
            f"context_length mismatch: stored={stored_cl}  actual={actual_cl}"
        )
    return f"context_length={actual_cl}"


def _no_empty_sequences() -> str:
    data  = torch.load(PROCESSED_FILE, map_location="cpu", weights_only=True)
    ids   = data["input_ids"]
    zeros = (ids.sum(dim=1) == 0).sum().item()
    if zeros > 0:
        raise ValueError(f"{zeros} all-zero sequences detected.")
    return f"all {len(ids):,} sequences are non-empty"


def _token_ids_in_range() -> str:
    data       = torch.load(PROCESSED_FILE, map_location="cpu", weights_only=True)
    vocab_size = data["vocab_size"]
    ids        = data["input_ids"]
    lo, hi     = int(ids.min()), int(ids.max())
    if lo < 0:
        raise ValueError(f"Negative token ID found: min={lo}")
    if hi >= vocab_size:
        raise ValueError(f"Token ID {hi} >= vocab_size {vocab_size}")
    return f"all IDs in [0, {vocab_size})  min={lo}  max={hi}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    checks: list[Check] = [
        _check("corpus exists",                _corpus_exists),
        _check("tokenizer exists",             _tokenizer_exists),
        _check("processed dataset exists",     _processed_exists),
        _check("input/target shapes match",    _shapes_match),
        _check("sequence lengths correct",     _seq_lengths),
        _check("no empty sequences",           _no_empty_sequences),
        _check("token IDs within vocab range", _token_ids_in_range),
    ]

    sep = "=" * 60
    print(sep)
    print("  MiniText8 Pipeline Verification")
    print(sep)

    all_passed = True
    for name, passed, detail in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"  [{icon}]  {name:<36}  {detail}")
        if not passed:
            all_passed = False

    print(sep)
    if all_passed:
        print("  Result: ALL CHECKS PASSED ✓")
    else:
        print("  Result: SOME CHECKS FAILED ✗")
    print(sep)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
