"""
scripts/train.py
================
Full MiniText8 training pipeline for FinanceLM.

Usage
-----
    python scripts/train.py
    python scripts/train.py --resume checkpoints/latest.pt
    python scripts/train.py --resume checkpoints/latest.pt --epochs 20

Arguments
---------
    --resume PATH       Resume from a checkpoint file.
    --epochs N          Override total epochs from config.
    --batch-size N      Override batch size from config.
    --lr FLOAT          Override learning rate from config.
    --context-length N  Override context length.
    --stride N          Override stride.
    --seed N            Random seed (default: 42).
    --log-level LEVEL   Logging verbosity (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from financelm.data.tokenizer_dataset import create_processed_dataloader
from financelm.model.config import ModelConfig
from financelm.model.model import FinanceLM
import financelm.paths as _fpaths
from financelm.paths import TRAINING_CONFIG
from financelm.training.checkpoint import CheckpointManager
from financelm.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_training_cfg() -> dict:
    try:
        import yaml
        with TRAINING_CONFIG.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _resolve_paths(cfg: dict) -> tuple[Path, Path, Path]:
    """
    Resolve PROCESSED_FILE, TOKENIZER_FILE, CHECKPOINT_DIR.

    Priority (highest first):
        1. configs/training.yaml  paths: section
        2. Environment variables  FINANCELM_*
        3. Project-relative defaults in financelm/paths.py
    """
    path_cfg = cfg.get("paths", {})

    def _pick(yaml_key: str, default: Path) -> Path:
        val = (path_cfg.get(yaml_key) or "").strip()
        return Path(val) if val else default

    processed  = _pick("processed_file",  _fpaths.PROCESSED_FILE)
    tokenizer  = _pick("tokenizer_file",  _fpaths.TOKENIZER_FILE)
    checkpoint = _pick("checkpoint_dir",  _fpaths.CHECKPOINT_DIR)
    return processed, tokenizer, checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)


def _warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min: float = 1e-5,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup followed by cosine decay.

    Returns a LambdaLR scheduler that can be stepped every optimiser step.
    """
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr  = float(optimizer.param_groups[0]["initial_lr"])
        return max(eta_min / base_lr, cosine)

    # store initial_lr so the lambda can read it
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FinanceLM on MiniText8.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch-size",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--context-length",type=int,   default=None)
    p.add_argument("--stride",        type=int,   default=None)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ── pre-flight checks ─────────────────────────────────────────────
    for path, hint in [
        (PROCESSED_FILE,  "python scripts/prepare_dataset.py"),
        (TOKENIZER_FILE,  "python scripts/train_tokenizer.py"),
    ]:
        if not path.exists():
            logger.error("Not found: %s  →  run: %s", path, hint)
            sys.exit(1)

    # ── config ────────────────────────────────────────────────────────
    cfg_yaml = _load_training_cfg()
    tr       = cfg_yaml.get("training",   {})
    ck       = cfg_yaml.get("checkpoint", {})

    PROCESSED_FILE, TOKENIZER_FILE, CHECKPOINT_DIR = _resolve_paths(cfg_yaml)
    gn       = cfg_yaml.get("generation", {})

    config = ModelConfig()
    config.max_seq_length = args.context_length or tr.get("context_length", 256)
    config.batch_size     = args.batch_size     or tr.get("batch_size",     8)
    config.epochs         = args.epochs         or tr.get("epochs",         10)
    config.learning_rate  = float(args.lr or tr.get("learning_rate", 3e-4))

    weight_decay     = float(tr.get("weight_decay",   0.1))
    warmup_steps     = int(  tr.get("warmup_steps",   1000))
    gradient_clip    = float(tr.get("gradient_clip",  1.0))
    val_fraction     = float(tr.get("val_fraction",   0.05))
    num_workers      = int(  tr.get("num_workers",    0))
    pin_memory       = bool( tr.get("pin_memory",     True))
    stride           = int(args.stride or tr.get("stride", 128))
    save_every_steps = int(  ck.get("save_every_steps", 500))
    seed             = args.seed

    _set_seed(seed)

    # ── device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model ─────────────────────────────────────────────────────────
    model = FinanceLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ── dataloaders ───────────────────────────────────────────────────
    train_loader, val_loader = create_processed_dataloader(
        processed_path=PROCESSED_FILE,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        val_fraction=val_fraction,
        seed=seed,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * config.epochs

    # ── optimiser & scheduler ─────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    scheduler = _warmup_cosine_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # ── checkpoint manager ────────────────────────────────────────────
    manager = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip=gradient_clip,
    )

    # ── resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_loss   = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        ckpt = manager.load(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            device=device,
        )
        start_epoch          = ckpt.get("epoch", 0)
        best_loss            = ckpt.get("loss",  float("inf"))
        trainer.global_step  = ckpt.get("global_step", 0)
        logger.info(
            "Resumed from epoch %d  (global_step=%d)",
            start_epoch, trainer.global_step,
        )

    # ── summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Script        : train.py")
    logger.info("Device        : %s", device)
    logger.info("Parameters    : %s", f"{total_params:,}")
    logger.info("Epochs        : %d  (start=%d)", config.epochs, start_epoch)
    logger.info("Batch size    : %d", config.batch_size)
    logger.info("Context len   : %d", config.max_seq_length)
    logger.info("Stride        : %d", stride)
    logger.info("LR            : %g", config.learning_rate)
    logger.info("Warmup steps  : %d", warmup_steps)
    logger.info("Total steps   : %d", total_steps)
    logger.info("Train batches : %d/epoch", steps_per_epoch)
    logger.info("Val batches   : %d/epoch", len(val_loader))
    logger.info("Save interval : every %d steps", save_every_steps)
    logger.info("=" * 60)

    # ── training loop ─────────────────────────────────────────────────
    t0 = time.perf_counter()

    for epoch in range(start_epoch, config.epochs):
        train_loss = trainer.train_epoch(train_loader, epoch=epoch + 1)
        val_loss   = trainer.validate_epoch(val_loader, epoch=epoch + 1)
        lr_now     = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d/%d  train=%.4f  val=%.4f  lr=%.2e  step=%d",
            epoch + 1, config.epochs,
            train_loss, val_loss, lr_now,
            trainer.global_step,
        )

        # per-epoch checkpoint
        manager.save(
            model=model, optimizer=optimizer,
            epoch=epoch + 1, loss=train_loss, config=config,
            scheduler=scheduler, scaler=trainer.scaler,
            global_step=trainer.global_step,
        )
        manager.save_latest(
            model=model, optimizer=optimizer,
            epoch=epoch + 1, loss=train_loss, config=config,
            scheduler=scheduler, scaler=trainer.scaler,
            global_step=trainer.global_step,
        )
        if train_loss < best_loss:
            best_loss = train_loss
            manager.save_best(
                model=model, optimizer=optimizer,
                epoch=epoch + 1, loss=train_loss, config=config,
                scheduler=scheduler, scaler=trainer.scaler,
                global_step=trainer.global_step,
            )
            logger.info("New best checkpoint  loss=%.4f", best_loss)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("Training complete  |  best_loss=%.4f  |  elapsed=%.0fs",
                best_loss, elapsed)
    logger.info("Checkpoints in: %s", CHECKPOINT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
