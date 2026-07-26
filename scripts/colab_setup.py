"""
scripts/colab_setup.py
=======================
Environment validation script for Google Colab.

Checks Python version, PyTorch installation, CUDA availability, GPU info,
and that all required pipeline artefacts are present.

Does NOT install packages — run  ``pip install -r requirements.txt``  first.

Usage
-----
    python scripts/colab_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── make sure the package is importable ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _section(title: str) -> None:
    print(f"\n{'=' * 56}")
    print(f"  {title}")
    print("=" * 56)


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [OK]  {label}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f"  — {detail}" if detail else ""
    print(f"  [!!]  {label}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  — {detail}" if detail else ""
    print(f"  [FAIL] {label}{suffix}")


def check_python() -> bool:
    _section("Python")
    major, minor = sys.version_info[:2]
    version_str  = f"{major}.{minor}.{sys.version_info.micro}"
    if (major, minor) >= (3, 10):
        _ok("Python version", version_str)
        return True
    _fail("Python version", f"{version_str} — need ≥ 3.10")
    return False


def check_torch() -> bool:
    _section("PyTorch")
    try:
        import torch
        _ok("torch installed", torch.__version__)
    except ImportError:
        _fail("torch not installed", "pip install torch")
        return False

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / (1024 ** 3)
            _ok(f"GPU {i}", f"{props.name}  {vram:.1f} GB VRAM")
        _ok("AMP (mixed precision)", "enabled for CUDA")
    else:
        _warn("CUDA not available", "training will run on CPU — much slower")
        _warn("AMP disabled", "CPU does not support float16 autocast")

    return True


def check_tokenizers() -> bool:
    _section("tokenizers library")
    try:
        import tokenizers
        _ok("tokenizers installed", tokenizers.__version__)
        return True
    except ImportError:
        _fail("tokenizers not installed", "pip install tokenizers")
        return False


def check_dependencies() -> bool:
    _section("Other dependencies")
    ok = True
    for pkg in ("tqdm", "yaml", "numpy"):
        try:
            __import__(pkg)
            _ok(pkg)
        except ImportError:
            _fail(pkg, f"pip install {pkg}")
            ok = False
    return ok


def check_artefacts() -> bool:
    _section("Pipeline artefacts")
    from financelm.paths import (
        PROCESSED_FILE,
        TOKENIZER_FILE,
        CHECKPOINT_DIR,
        DATASET_CONFIG,
        TRAINING_CONFIG,
    )
    from pathlib import Path
    project_root    = Path(__file__).resolve().parent.parent
    combined_corpus = project_root / "datasets" / "combined" / "corpus.txt"
    minitext8_file  = project_root / "data" / "minitext8" / "minitext8.txt"

    ok = True

    # ── Required artefacts ───────────────────────────────────────────
    required = [
        (TOKENIZER_FILE,  "Tokenizer (tokenizer.json)", "python scripts/train_tokenizer.py"),
        (PROCESSED_FILE,  "Processed dataset (.pt)",    "python scripts/prepare_dataset.py"),
        (DATASET_CONFIG,  "configs/dataset.yaml",       None),
        (TRAINING_CONFIG, "configs/training.yaml",      None),
    ]
    for path, label, hint in required:
        if path.exists():
            size = path.stat().st_size
            _ok(label, f"{size:,} bytes")
        else:
            msg = f"Run:  {hint}" if hint else f"Missing: {path}"
            _fail(label, msg)
            ok = False

    # ── Corpus — combined preferred, MiniText8 acceptable ────────────
    if combined_corpus.exists():
        size = combined_corpus.stat().st_size
        _ok("Combined corpus", f"{size:,} bytes")
    elif minitext8_file.exists():
        size = minitext8_file.stat().st_size
        _warn("Combined corpus not found", "run: python scripts/build_corpus.py")
        _ok("MiniText8 corpus (fallback)", f"{size:,} bytes")
    else:
        _fail("Corpus", "run: python scripts/download_minitext8.py then build_corpus.py")
        ok = False

    # ── Checkpoints — optional before first training run ─────────────
    latest = CHECKPOINT_DIR / "latest.pt"
    if latest.exists():
        _ok("Latest checkpoint", str(latest))
    else:
        _warn("No checkpoint found", "training will start from scratch")

    return ok


def check_configs() -> bool:
    _section("Configuration")
    from financelm.paths import TRAINING_CONFIG, DATASET_CONFIG
    try:
        import yaml
        for cfg_path in (DATASET_CONFIG, TRAINING_CONFIG):
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            _ok(cfg_path.name, f"{len(data)} top-level keys")
        return True
    except Exception as exc:
        _fail("Config load failed", str(exc))
        return False


def main() -> None:
    print("=" * 56)
    print("  FinanceLM — Colab Environment Check")
    print("=" * 56)

    results = [
        check_python(),
        check_torch(),
        check_tokenizers(),
        check_dependencies(),
        check_artefacts(),
        check_configs(),
    ]

    _section("Summary")
    passed = sum(results)
    total  = len(results)
    if all(results):
        print(f"  All {total} checks passed ✓  —  ready to train!")
    else:
        failed = total - passed
        print(f"  {passed}/{total} checks passed  |  {failed} issue(s) to resolve.")
    print("=" * 56)

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
