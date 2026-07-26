"""
scripts/evaluate.py
====================
Evaluates a trained FinanceLM checkpoint on the MiniText8 dataset.

Reports
-------
- Loss and perplexity on the full processed dataset (or a held-out split)
- Sample generations from fixed prompts
- Model parameter count
- Checkpoint metadata

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val

Arguments
---------
    --checkpoint PATH   Checkpoint to evaluate (default: checkpoints/latest.pt).
    --split SPLIT       Which split to evaluate on: train | val | all (default: all).
    --val-fraction F    Val fraction matching prepare_dataset split (default: 0.05).
    --seed N            Seed for reproducible val split (default: 42).
    --batch-size N      Evaluation batch size (default: 16).
    --num-workers N     DataLoader workers (default: 0).
    --log-level LEVEL   Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from financelm.data.tokenizer_dataset import (
    ProcessedDataset,
    split_dataset,
)
from financelm.inference.generator import Generator
from financelm.inference.sampling import SamplingStrategy
from financelm.model.config import ModelConfig
from financelm.model.model import FinanceLM
from financelm.paths import CHECKPOINT_DIR, PROCESSED_FILE, TOKENIZER_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")

_SAMPLE_PROMPTS = [
    "anarchism originated as a term of",
    "the stock market",
    "mathematics is the study of",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a FinanceLM checkpoint.")
    p.add_argument("--checkpoint",    type=str,   default=None)
    p.add_argument("--split",         type=str,   default="all",
                   choices=["train", "val", "all"])
    p.add_argument("--val-fraction",  type=float, default=0.05)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


@torch.no_grad()
def compute_loss_perplexity(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> tuple[float, float]:
    """
    Compute mean cross-entropy loss and perplexity over *loader*.

    Returns
    -------
    (loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    n_batches  = len(loader)

    for input_ids, target_ids in tqdm(loader, desc=f"Eval [{label}]",
                                       leave=False, dynamic_ncols=True):
        input_ids  = input_ids.to(device,  non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        logits = model(input_ids)
        loss   = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )
        total_loss += loss.item()

    mean_loss  = total_loss / n_batches
    perplexity = math.exp(mean_loss)
    return mean_loss, perplexity


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── pre-flight ────────────────────────────────────────────────────
    for path, hint in [
        (PROCESSED_FILE, "python scripts/prepare_dataset.py"),
        (TOKENIZER_FILE, "python scripts/train_tokenizer.py"),
    ]:
        if not path.exists():
            logger.error("Not found: %s  →  run: %s", path, hint)
            sys.exit(1)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        CHECKPOINT_DIR / "latest.pt"
    )
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        logger.error("Run:  python scripts/train.py")
        sys.exit(1)

    # ── model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the raw checkpoint payload first so we can read the saved config
    # before constructing the model. This ensures evaluation always uses
    # the exact same architecture that was used during training.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_cfg = ckpt.get("config", {})
    config = ModelConfig(
        vocab_size       = saved_cfg.get("vocab_size",       ModelConfig.vocab_size),
        embedding_dim    = saved_cfg.get("embedding_dim",    ModelConfig.embedding_dim),
        num_heads        = saved_cfg.get("num_heads",        ModelConfig.num_heads),
        num_layers       = saved_cfg.get("num_layers",       ModelConfig.num_layers),
        feed_forward_dim = saved_cfg.get("feed_forward_dim", ModelConfig.feed_forward_dim),
        max_seq_length   = saved_cfg.get("max_seq_length",   ModelConfig.max_seq_length),
        dropout          = saved_cfg.get("dropout",          ModelConfig.dropout),
    ) if saved_cfg else ModelConfig()

    model = FinanceLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # ── dataset split ─────────────────────────────────────────────────
    dataset = ProcessedDataset(PROCESSED_FILE)
    train_sub, val_sub = split_dataset(dataset, args.val_fraction, args.seed)

    def _loader(subset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    # ── evaluation ────────────────────────────────────────────────────
    results: dict[str, tuple[float, float]] = {}
    t0 = time.perf_counter()

    if args.split in ("train", "all"):
        results["train"] = compute_loss_perplexity(
            model, _loader(train_sub), device, "train"
        )

    if args.split in ("val", "all"):
        results["val"] = compute_loss_perplexity(
            model, _loader(val_sub), device, "val"
        )

    elapsed = time.perf_counter() - t0

    # ── sample generations ────────────────────────────────────────────
    generator = Generator(
        model=model,
        tokenizer_path=TOKENIZER_FILE,
        device=device,
        max_seq_length=config.max_seq_length,
    )
    samples: list[tuple[str, str]] = []
    for prompt in _SAMPLE_PROMPTS:
        try:
            text = generator.generate(
                prompt=prompt,
                max_new_tokens=80,
                strategy=SamplingStrategy.TOP_P,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
            )
            samples.append((prompt, text))
        except Exception as exc:
            samples.append((prompt, f"[generation failed: {exc}]"))

    # ── report ────────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("  FinanceLM Evaluation Report")
    print(sep)
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"  Epoch         : {ckpt.get('epoch', '?')}")
    print(f"  Global step   : {ckpt.get('global_step', '?')}")
    print(f"  Training loss : {ckpt.get('loss', float('nan')):.4f}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  Device        : {device}")
    print(f"  Elapsed       : {elapsed:.1f}s")
    print()
    print("  Metrics")
    print("  " + "-" * 40)
    for split, (loss, ppl) in results.items():
        print(f"  {split:<8}  loss={loss:.4f}  perplexity={ppl:,.1f}")
    print()
    print("  Sample Generations")
    print("  " + "-" * 40)
    for prompt, output in samples:
        print(f"  Prompt : {prompt!r}")
        print(f"  Output : {output!r}")
        print()
    print(sep)


if __name__ == "__main__":
    main()
