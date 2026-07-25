"""
financelm.data.loader
=====================
Loads the MiniText8 corpus from disk and returns it as a plain Python string.

MiniText8 is a 100 MB subset of the enwik9 Wikipedia XML dump, preprocessed
to contain only lowercase ASCII text.  It is used here to validate the full
FinanceLM training pipeline end-to-end before switching to the real dataset.

The loader is intentionally simple: it reads the file, validates it, and
returns the raw text.  All further processing (whitespace normalisation,
tokenization, windowing) is handled by separate modules.

Usage
-----
    from financelm.data.loader import load_minitext8
    text = load_minitext8(Path("data/minitext8/minitext8.txt"))
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_minitext8(path: Path) -> str:
    """
    Read the MiniText8 corpus file and return its content as a string.

    Parameters
    ----------
    path:
        Path to the ``minitext8.txt`` file.

    Returns
    -------
    str
        Full text content of the corpus.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    UnicodeDecodeError
        If the file is not valid UTF-8.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"MiniText8 corpus not found: {path}\n"
            "Run:  python scripts/download_minitext8.py"
        )

    logger.info("Loading corpus from %s", path)
    text = path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"Corpus file is empty: {path}")

    logger.info("Loaded %d characters", len(text))
    return text
