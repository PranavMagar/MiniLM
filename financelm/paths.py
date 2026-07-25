"""
financelm.paths
===============
Canonical project-wide path constants, resolved relative to the project
root so every script works regardless of the current working directory.

Override via environment variables
-----------------------------------
Any path can be overridden at runtime without changing source code:

    FINANCELM_CHECKPOINT_DIR=/content/drive/MyDrive/FinanceLM/checkpoints
    FINANCELM_PROCESSED_FILE=/content/drive/MyDrive/FinanceLM/tokenized.pt
    FINANCELM_TOKENIZER_FILE=/content/drive/MyDrive/FinanceLM/tokenizer.json
    FINANCELM_MINITEXT8_FILE=/content/drive/MyDrive/FinanceLM/minitext8.txt

This is the recommended approach for Google Colab with mounted Drive.
Alternatively, set the ``paths:`` section in ``configs/training.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — always the parent of this file's directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(var: str, default: Path) -> Path:
    """Return ``default`` unless the environment variable ``var`` is set."""
    val = os.environ.get(var, "").strip()
    return Path(val) if val else default


# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data"
MINITEXT8_DIR = DATA_DIR / "minitext8"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
PROCESSED_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Key files  (overridable via env vars)
# ---------------------------------------------------------------------------
MINITEXT8_FILE = _env("FINANCELM_MINITEXT8_FILE", MINITEXT8_DIR / "minitext8.txt")
TOKENIZER_FILE = _env("FINANCELM_TOKENIZER_FILE", TOKENIZER_DIR / "tokenizer.json")
PROCESSED_FILE = _env("FINANCELM_PROCESSED_FILE", PROCESSED_DIR / "tokenized.pt")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIGS_DIR     = PROJECT_ROOT / "configs"
DATASET_CONFIG  = CONFIGS_DIR / "dataset.yaml"
TRAINING_CONFIG = CONFIGS_DIR / "training.yaml"

# ---------------------------------------------------------------------------
# Checkpoints  (overridable via env var)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = _env("FINANCELM_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")

# ---------------------------------------------------------------------------
# Auto-create directories on import
# ---------------------------------------------------------------------------
for _d in (MINITEXT8_DIR, TOKENIZER_DIR, PROCESSED_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
