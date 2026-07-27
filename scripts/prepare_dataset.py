"""
scripts/prepare_dataset.py
===========================
Streaming preprocessing pipeline for FinanceLM.

Architecture
------------
The corpus is processed in fixed-size text chunks rather than being read
into memory all at once.  This keeps RAM usage nearly constant regardless
of corpus size and scales from 2 GB to hundreds of GB without changes.

Streaming pipeline per chunk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Read 8 MB text chunk
       │
       ▼
  tokenizer.encode(chunk)          ← never tokenize the full corpus at once
       │
       ▼
  Extend rolling token buffer      ← list of ints, bounded in size
       │
       ▼
  Drain complete windows           ← slide by stride, emit (input, target)
       │
       ▼
  Write to memory-mapped array     ← no Python objects per window
       │
       ▼
  Discard processed tokens         ← keep only the overlap needed for next chunk
       │
       ▼
  Repeat until EOF

Memory usage
------------
At any moment only two things are in RAM:
  - The current 8 MB text chunk (constant)
  - The rolling token buffer, at most CHUNK_CHARS / avg_chars_per_token tokens
    (~2–3 MB for 8 MB text)

The pre-allocated memmap array lives on disk.  The final torch.save step
reads the memmap back in one pass to write the .pt file, so peak RAM at
that point is proportional to the output size — the same as before.

Output format  (unchanged — compatible with ProcessedDataset and train.py)
----------------------------------------------------------------------------
    {
        "input_ids":      LongTensor  shape (N, context_length),
        "target_ids":     LongTensor  shape (N, context_length),
        "vocab_size":     int,
        "context_length": int,
        "stride":         int,
        "n_tokens":       int,
        "corpus":         str,
    }

Usage
-----
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --corpus data/minitext8/minitext8.txt
    python scripts/prepare_dataset.py --context-length 512 --stride 256
    python scripts/prepare_dataset.py --chunk-mb 16

Arguments
---------
    --corpus          Corpus file (default: datasets/combined/corpus.txt).
    --context-length  Tokens per window (default: from configs/dataset.yaml).
    --stride          Step between windows (default: from config).
    --chunk-mb        Text read per iteration in MB (default: 8).
    --log-level       Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        import yaml  # type: ignore[import]
        with DATASET_CONFIG.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Streaming tokenizer
# ---------------------------------------------------------------------------

def _estimate_windows(file_size: int, chunk_bytes: int, context_length: int,
                      stride: int, chars_per_token: float = 4.5) -> int:
    """
    Conservative upper bound on window count for memmap pre-allocation.

    Uses chars_per_token to estimate total tokens; the actual count may be
    lower (e.g. after the tokenizer merges subwords).
    """
    estimated_tokens = int(file_size / chars_per_token)
    if estimated_tokens < context_length + 1:
        return 0
    return (estimated_tokens - context_length) // stride + 1


def stream_tokenize_windows(
    corpus_path: Path,
    tokenizer: "Tokenizer",
    context_length: int,
    stride: int,
    chunk_bytes: int,
    memmap_path: Path,
) -> tuple[int, int]:
    """
    Stream *corpus_path* in chunks, tokenize each chunk, emit sliding windows
    into a memory-mapped array on disk.

    Parameters
    ----------
    corpus_path:
        Path to the plain-text corpus.
    tokenizer:
        Loaded HuggingFace tokenizers ``Tokenizer`` instance.
    context_length:
        Tokens per window.
    stride:
        Step between consecutive windows.
    chunk_bytes:
        Number of bytes read per iteration.
    memmap_path:
        Path of the pre-allocated ``numpy.memmap`` array to write into.
        Shape: ``(max_windows, context_length + 1)`` — the +1 column stores
        both input ([:context_length]) and target ([1:context_length+1]) in
        a single pass; we split on load.

    Returns
    -------
    (n_windows, n_tokens) : tuple[int, int]
        Actual number of windows written and total tokens processed.
    """
    file_size     = corpus_path.stat().st_size
    max_windows   = _estimate_windows(file_size, chunk_bytes, context_length, stride)

    # Pre-allocate memory-mapped array on disk.
    # Each row stores context_length+1 tokens: [t0, t1, ..., t_CL]
    # input_ids  = row[:context_length]
    # target_ids = row[1:]
    # Remove any stale temp file from a previous failed run first.
    if Path(str(memmap_path)).exists():
        try:
            Path(str(memmap_path)).unlink()
        except PermissionError:
            pass  # Windows may still hold a lock; numpy will overwrite anyway

    mm = np.memmap(
        str(memmap_path),
        dtype=np.int32,
        mode="w+",
        shape=(max_windows, context_length + 1),
    )

    buffer: list[int] = []          # rolling token buffer
    n_windows    = 0
    n_tokens     = 0
    t0           = time.perf_counter()

    pbar = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Streaming",
        dynamic_ncols=True,
        smoothing=0.05,
    )

    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            # ── read one chunk ────────────────────────────────────────
            raw = fh.read(chunk_bytes)
            if not raw:
                break

            bytes_read = len(raw.encode("utf-8", errors="replace"))
            pbar.update(bytes_read)

            # ── tokenize the chunk ───────────────────────────────────
            # encode() handles the chunk as a self-contained string.
            # Subword context doesn't carry across chunks, but for BPE
            # on whitespace-pre-tokenized text this is acceptable — the
            # chunk boundary may split a word at most once per 8 MB.
            new_tokens: list[int] = tokenizer.encode(raw).ids
            buffer.extend(new_tokens)
            n_tokens += len(new_tokens)

            # ── drain complete windows ────────────────────────────────
            # We need context_length+1 tokens to form one (input, target) pair.
            while len(buffer) >= context_length + 1:
                if n_windows >= max_windows:
                    # Pre-allocation was too conservative; grow the memmap.
                    new_max = max_windows + max(10_000, max_windows // 4)
                    logger.debug(
                        "Growing memmap from %d to %d windows", max_windows, new_max
                    )
                    mm.flush()
                    del mm
                    mm = np.memmap(
                        str(memmap_path),
                        dtype=np.int32,
                        mode="r+",
                        shape=(new_max, context_length + 1),
                    )
                    max_windows = new_max

                # Write context_length+1 tokens as one row
                mm[n_windows, :] = buffer[: context_length + 1]
                n_windows += 1

                # Advance buffer by stride
                del buffer[:stride]

            # ── progress postfix ──────────────────────────────────────
            elapsed = time.perf_counter() - t0
            speed   = n_tokens / elapsed if elapsed > 0 else 0
            pbar.set_postfix(
                tokens=f"{n_tokens/1e6:.1f}M",
                windows=f"{n_windows:,}",
                ktok_s=f"{speed/1e3:.0f}k",
                refresh=False,
            )

    pbar.close()

    # Flush any remaining buffer tokens into a final partial window
    # (only if enough tokens remain to form a full window)
    while len(buffer) >= context_length + 1:
        if n_windows >= max_windows:
            new_max = max_windows + 10_000
            mm.flush(); del mm
            mm = np.memmap(
                str(memmap_path), dtype=np.int32, mode="r+",
                shape=(new_max, context_length + 1),
            )
            max_windows = new_max
        mm[n_windows, :] = buffer[: context_length + 1]
        n_windows += 1
        del buffer[:stride]

    mm.flush()
    del mm
    return n_windows, n_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream-tokenize a corpus and build a sliding-window dataset."
    )
    p.add_argument("--corpus", type=str, default=None,
                   help="Corpus file (default: datasets/combined/corpus.txt).")
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--stride",         type=int, default=None)
    p.add_argument("--chunk-mb",       type=float, default=8.0,
                   help="Text chunk size in MB per iteration (default: 8).")
    p.add_argument("--log-level",      default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── resolve corpus path ───────────────────────────────────────────
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

    # ── config ───────────────────────────────────────────────────────
    cfg            = _load_config().get("training", {})
    context_length = int(args.context_length or cfg.get("context_length", 256))
    stride         = int(args.stride         or cfg.get("stride",         128))
    chunk_bytes    = int(args.chunk_mb * (1 << 20))

    corpus_mb  = corpus_path.stat().st_size / (1 << 20)
    tokenizer  = Tokenizer.from_file(str(TOKENIZER_FILE))
    vocab_size = tokenizer.get_vocab_size()

    logger.info("=" * 60)
    logger.info("Script         : prepare_dataset.py  (streaming)")
    logger.info("Corpus         : %s (%.0f MB)", corpus_path, corpus_mb)
    logger.info("Tokenizer      : %s  (vocab=%d)", TOKENIZER_FILE, vocab_size)
    logger.info("Context length : %d", context_length)
    logger.info("Stride         : %d", stride)
    logger.info("Chunk size     : %.0f MB", args.chunk_mb)
    logger.info("Output         : %s", PROCESSED_FILE)
    logger.info("=" * 60)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── stream-tokenize into a temp memmap ───────────────────────────
    # Use a temp file so the output is either complete or absent —
    # a partial run never leaves a corrupt .pt behind.
    tmp_dir  = PROCESSED_FILE.parent
    tmp_mm   = tmp_dir / "_tokenized_tmp.mmap"

    try:
        t0 = time.perf_counter()
        n_windows, n_tokens = stream_tokenize_windows(
            corpus_path    = corpus_path,
            tokenizer      = tokenizer,
            context_length = context_length,
            stride         = stride,
            chunk_bytes    = chunk_bytes,
            memmap_path    = tmp_mm,
        )
        stream_elapsed = time.perf_counter() - t0

        if n_windows == 0:
            logger.error("No windows produced — corpus too small or context_length too large.")
            sys.exit(1)

        logger.info(
            "Streaming complete in %.1fs  |  tokens=%d  windows=%d",
            stream_elapsed, n_tokens, n_windows,
        )

        # ── load memmap → tensors → save .pt ─────────────────────────
        # This is the only step that temporarily holds the full dataset
        # in RAM — same peak as before, but only at this final stage.
        logger.info("Writing %s …", PROCESSED_FILE)
        t1 = time.perf_counter()

        mm = np.memmap(str(tmp_mm), dtype=np.int32, mode="r",
                       shape=(n_windows, context_length + 1))

        # Convert the memmap slice to a contiguous int64 tensor in one shot
        data = torch.from_numpy(np.array(mm[:n_windows], dtype=np.int64))

        input_ids  = data[:, :context_length]        # (N, L)
        target_ids = data[:, 1:]                     # (N, L)  — offset by 1

        payload = {
            "input_ids":      input_ids.contiguous(),
            "target_ids":     target_ids.contiguous(),
            "vocab_size":     vocab_size,
            "context_length": context_length,
            "stride":         stride,
            "n_tokens":       n_tokens,
            "corpus":         str(corpus_path),
        }

        del mm, data   # release memmap reference before overwriting

        torch.save(payload, PROCESSED_FILE)
        save_elapsed = time.perf_counter() - t1

    finally:
        # Clean up the temp memmap.
        # On Windows, the file must be fully dereferenced before unlinking.
        import gc
        gc.collect()
        if tmp_mm.exists():
            try:
                tmp_mm.unlink()
            except PermissionError:
                # Windows: file still held by a previous memmap reference.
                # Not a problem — it will be cleaned up on next run or OS reboot.
                logger.debug("Could not delete temp memmap (Windows lock): %s", tmp_mm)

    total_elapsed = time.perf_counter() - t0
    output_mb     = PROCESSED_FILE.stat().st_size / (1 << 20)

    logger.info("=" * 60)
    logger.info("Windows        : %d", n_windows)
    logger.info("Total tokens   : %d", n_tokens)
    logger.info("Output size    : %.1f MB", output_mb)
    logger.info("Stream time    : %.1fs", stream_elapsed)
    logger.info("Save time      : %.1fs", save_elapsed)
    logger.info("Total time     : %.1fs", total_elapsed)
    logger.info("=" * 60)
    logger.info("Done.  Run scripts/verify_dataset.py to validate.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
