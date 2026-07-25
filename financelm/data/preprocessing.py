"""
financelm.data.preprocessing
=============================
Cleans and normalises raw MiniText8 text before tokenization.

Rules applied
-------------
- Validate UTF-8 encoding (re-encode/decode round-trip).
- Normalise line endings to ``\\n``.
- Collapse sequences of more than two consecutive blank lines to two.
- Collapse runs of spaces/tabs on a single line to a single space.
- Strip leading and trailing whitespace from each line.
- Remove completely empty samples after the above steps.
- Preserve punctuation, capitalisation, and numbers.

What is NOT done
----------------
- No lowercasing.
- No punctuation removal.
- No stemming / lemmatisation.

Usage
-----
    from financelm.data.preprocessing import preprocess
    clean = preprocess(raw_text)
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_RE_CRLF        = re.compile(r"\r\n?")
_RE_HSPACE      = re.compile(r"[ \t]+")
_RE_BLANK_LINES = re.compile(r"\n{3,}")


def preprocess(text: str) -> str:
    """
    Apply lightweight normalisation to raw corpus text.

    Parameters
    ----------
    text:
        Raw corpus string.

    Returns
    -------
    str
        Normalised corpus string.

    Raises
    ------
    ValueError
        If *text* is empty after normalisation.
    """
    if not text:
        raise ValueError("Input text is empty.")

    # Validate UTF-8 by round-tripping through bytes
    try:
        text = text.encode("utf-8").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"UTF-8 validation failed: {exc}") from exc

    original_len = len(text)

    # Normalise line endings
    text = _RE_CRLF.sub("\n", text)

    # Per-line: collapse horizontal whitespace and strip edges
    lines = [_RE_HSPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse excessive blank lines
    text = _RE_BLANK_LINES.sub("\n\n", text)

    # Final strip
    text = text.strip()

    if not text:
        raise ValueError("Text is empty after preprocessing.")

    logger.info(
        "Preprocessing: %d → %d chars (%.1f%% retained)",
        original_len,
        len(text),
        100.0 * len(text) / original_len,
    )
    return text
