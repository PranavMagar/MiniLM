"""
scripts/build_corpus.py
========================
Combines multiple plain-text datasets into a single corpus file for
tokenizer training and language model pretraining.

Corpus format
-------------
Documents are separated by a single ``<|endoftext|>`` token on its own line.
No other separators or metadata are inserted.

    <document 1>
    <|endoftext|>
    <document 2>
    <|endoftext|>
    ...

Adding new datasets
-------------------
Register a new source in the ``SOURCES`` list at the top of main().
Each source is a dict with:
    path   : Path to the raw text file
    mode   : "single"    — entire file is one document
              "paragraph" — blank-line-separated blocks are individual docs
    name   : Human-readable label for logging

Current sources (in order of concatenation)
--------------------------------------------
1. MiniText8   — single continuous Wikipedia text block
2. TinyStories — 2.1M short stories (blank-line-separated)

Usage
-----
    python scripts/build_corpus.py
    python scripts/build_corpus.py --output datasets/combined/corpus.txt
    python scripts/build_corpus.py --separator "<|endoftext|>"

Arguments
---------
    --output        Output file path (default: datasets/combined/corpus.txt).
    --separator     Token inserted between documents (default: <|endoftext|>).
    --min-chars     Drop documents shorter than N characters (default: 20).
    --no-tinystories  Exclude TinyStories from the corpus.
    --no-minitext8    Exclude MiniText8 from the corpus.
    --log-level     Logging verbosity (default: INFO).

Output
------
    datasets/combined/corpus.txt
    datasets/combined/corpus_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_corpus")

_RE_HSPACE      = re.compile(r"[ \t]+")
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_CRLF        = re.compile(r"\r\n?")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """
    Lightweight normalisation:
    - UTF-8 round-trip validation
    - NFC unicode normalisation
    - CRLF → LF
    - Collapse runs of spaces/tabs to a single space
    - Collapse 3+ consecutive blank lines to 2
    - Strip leading/trailing whitespace

    Does NOT lowercase, remove punctuation, or remove numbers.
    """
    text = unicodedata.normalize("NFC", text.encode("utf-8").decode("utf-8"))
    text = _RE_CRLF.sub("\n", text)
    lines = [_RE_HSPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Document iterators — one per source mode
# ---------------------------------------------------------------------------

def _iter_single(path: Path) -> list[str]:
    """
    The entire file is treated as one document.

    Used for MiniText8, which is a single continuous string.
    """
    text = path.read_text(encoding="utf-8")
    doc  = _clean(text)
    return [doc] if doc else []


def _iter_paragraphs(path: Path, min_chars: int) -> list[str]:
    """
    Split on blank lines (two or more consecutive newlines).
    Each non-empty block is one document.

    Used for TinyStories, which uses blank lines between stories.
    Streams the file in chunks to keep memory usage low.
    """
    docs: list[str] = []
    buffer: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = _RE_CRLF.sub("\n", raw_line)
            if line.strip() == "":
                # Blank line = document boundary
                if buffer:
                    doc = _clean("\n".join(buffer))
                    if doc and len(doc) >= min_chars:
                        docs.append(doc)
                    buffer = []
            else:
                buffer.append(line.rstrip("\n"))

    # Flush the last document
    if buffer:
        doc = _clean("\n".join(buffer))
        if doc and len(doc) >= min_chars:
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined training corpus.")
    p.add_argument("--output",         default="datasets/combined/corpus.txt")
    p.add_argument("--separator",      default="<|endoftext|>",
                   help="Token between documents (default: <|endoftext|>).")
    p.add_argument("--min-chars",      type=int, default=20,
                   help="Drop documents shorter than N chars (default: 20).")
    p.add_argument("--no-minitext8",   action="store_true")
    p.add_argument("--no-tinystories", action="store_true")
    p.add_argument("--log-level",      default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    root   = Path(__file__).resolve().parent.parent
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ------------------------------------------------------------------
    # Source registry
    # Add new datasets here — no other code changes required.
    # ------------------------------------------------------------------
    SOURCES = []
    if not args.no_minitext8:
        SOURCES.append({
            "name": "MiniText8",
            "path": root / "data" / "minitext8" / "minitext8.txt",
            "mode": "single",        # entire file = 1 document
        })
    if not args.no_tinystories:
        SOURCES.append({
            "name": "TinyStories-train",
            "path": root / "datasets" / "tinystories" / "train.txt",
            "mode": "paragraph",     # blank-line-separated stories
        })
    # Future datasets: add entries here, e.g.:
    # {"name": "FineWeb",  "path": ..., "mode": "paragraph"}
    # {"name": "Finance",  "path": ..., "mode": "paragraph"}

    if not SOURCES:
        logger.error("No sources selected. Remove --no-* flags.")
        sys.exit(1)

    sep = args.separator
    min_chars = args.min_chars

    logger.info("=" * 60)
    logger.info("Script     : build_corpus.py")
    logger.info("Output     : %s", output)
    logger.info("Separator  : %s", sep)
    logger.info("Min chars  : %d", min_chars)
    logger.info("Sources    : %s", [s["name"] for s in SOURCES])
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Validate all source paths up front
    # ------------------------------------------------------------------
    for src in SOURCES:
        if not src["path"].exists():
            logger.error("Source not found: %s  (%s)", src["path"], src["name"])
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build corpus
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    total_docs  = 0
    total_chars = 0
    total_words = 0
    source_stats: list[dict] = []

    with output.open("w", encoding="utf-8") as out_fh:
        for src in SOURCES:
            src_name  = src["name"]
            src_path  = src["path"]
            src_mode  = src["mode"]

            logger.info("Processing  %s  (mode=%s) …", src_name, src_mode)
            t1 = time.perf_counter()

            if src_mode == "single":
                docs = _iter_single(src_path)
            elif src_mode == "paragraph":
                docs = _iter_paragraphs(src_path, min_chars)
            else:
                logger.error("Unknown mode '%s' for source '%s'", src_mode, src_name)
                sys.exit(1)

            src_chars = 0
            src_words = 0
            src_docs  = 0

            for doc in docs:
                # Write separator before every document except the very first
                if total_docs > 0:
                    out_fh.write(f"\n{sep}\n")

                out_fh.write(doc)

                src_chars += len(doc)
                src_words += len(doc.split())
                src_docs  += 1
                total_docs += 1

            src_elapsed = time.perf_counter() - t1
            total_chars += src_chars
            total_words += src_words

            logger.info(
                "  %-20s  docs=%8d  chars=%12d  words=%9d  %.1fs",
                src_name, src_docs, src_chars, src_words, src_elapsed,
            )
            source_stats.append({
                "name":      src_name,
                "path":      str(src_path),
                "mode":      src_mode,
                "documents": src_docs,
                "characters": src_chars,
                "words":     src_words,
            })

    elapsed    = time.perf_counter() - t0
    file_size  = output.stat().st_size
    est_tokens = total_chars // 4   # ~4 chars/token for BPE

    stats = {
        "output":            str(output),
        "file_size_bytes":   file_size,
        "total_documents":   total_docs,
        "total_characters":  total_chars,
        "total_words":       total_words,
        "estimated_tokens":  est_tokens,
        "separator":         sep,
        "min_chars":         min_chars,
        "elapsed_s":         round(elapsed, 1),
        "sources":           source_stats,
    }

    stats_path = output.parent / "corpus_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("=" * 60)
    logger.info("Corpus saved : %s", output)
    logger.info("File size    : %.1f MB", file_size / (1 << 20))
    logger.info("Documents    : %d", total_docs)
    logger.info("Characters   : %d", total_chars)
    logger.info("Words        : %d", total_words)
    logger.info("Est. tokens  : %d", est_tokens)
    logger.info("Elapsed      : %.1fs", elapsed)
    logger.info("Stats saved  : %s", stats_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
