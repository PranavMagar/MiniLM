"""
financelm.training.checkpoint
==============================
Saves and loads model checkpoints with full training state.

Checkpoint payload
------------------
{
    "epoch":                int,
    "global_step":          int,
    "loss":                 float,
    "config":               dict,
    "model_state_dict":     dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "scaler_state_dict":    dict | None,
    "rng_state":            dict,
}

The rng_state dict stores Python, NumPy, and PyTorch random states so
training can resume with deterministic behaviour.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from financelm.paths import CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def _capture_rng() -> dict:
    """Capture current RNG states from Python, NumPy, and PyTorch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    """Restore RNG states captured by ``_capture_rng``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    """
    Saves and loads full training-state checkpoints.

    Parameters
    ----------
    checkpoint_dir:
        Directory to save checkpoints in.  Defaults to
        ``CHECKPOINT_DIR`` from ``financelm.paths``.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.checkpoint_dir    = Path(checkpoint_dir or CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint = self.checkpoint_dir / "latest.pt"
        self.best_checkpoint   = self.checkpoint_dir / "best.pt"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model:     Any,
        optimizer: Any,
        epoch:     int,
        loss:      float,
        config:    Any,
        scheduler: Any | None = None,
        scaler:    Any | None = None,
        global_step: int = 0,
        filename:  str | None = None,
    ) -> Path:
        """
        Save a full checkpoint.

        Parameters
        ----------
        model, optimizer:
            Model and optimiser whose state dicts are saved.
        epoch:
            Completed epoch number.
        loss:
            Epoch loss (used to track best).
        config:
            ModelConfig dataclass (converted to dict via ``vars``).
        scheduler:
            Optional LR scheduler.
        scaler:
            Optional AMP GradScaler.
        global_step:
            Total optimiser steps completed.
        filename:
            Override filename; defaults to ``epoch_{epoch}.pt``.

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        payload: dict[str, Any] = {
            "epoch":                epoch,
            "global_step":          global_step,
            "loss":                 loss,
            "config":               vars(config),
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict":    scaler.state_dict()    if scaler    else None,
            "rng_state":            _capture_rng(),
        }

        filename  = filename or f"epoch_{epoch}.pt"
        save_path = self.checkpoint_dir / filename
        torch.save(payload, save_path)
        logger.info("Checkpoint saved: %s", save_path)
        return save_path

    def save_latest(
        self, model: Any, optimizer: Any, epoch: int, loss: float, config: Any,
        scheduler: Any | None = None, scaler: Any | None = None, global_step: int = 0,
    ) -> Path:
        """Save as ``latest.pt``."""
        return self.save(
            model, optimizer, epoch, loss, config,
            scheduler=scheduler, scaler=scaler, global_step=global_step,
            filename="latest.pt",
        )

    def save_best(
        self, model: Any, optimizer: Any, epoch: int, loss: float, config: Any,
        scheduler: Any | None = None, scaler: Any | None = None, global_step: int = 0,
    ) -> Path:
        """Save as ``best.pt``."""
        return self.save(
            model, optimizer, epoch, loss, config,
            scheduler=scheduler, scaler=scaler, global_step=global_step,
            filename="best.pt",
        )

    def save_step(
        self, model: Any, optimizer: Any, epoch: int, loss: float, config: Any,
        scheduler: Any | None = None, scaler: Any | None = None, global_step: int = 0,
    ) -> Path:
        """Save a step-level checkpoint as ``step_{global_step}.pt``."""
        return self.save(
            model, optimizer, epoch, loss, config,
            scheduler=scheduler, scaler=scaler, global_step=global_step,
            filename=f"step_{global_step}.pt",
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: Path | str,
        model:     Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler:    Any | None = None,
        device:    str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> dict:
        """
        Load a checkpoint and restore model (and optionally optimiser /
        scheduler / scaler / RNG) state.

        Parameters
        ----------
        checkpoint_path:
            Path to the ``.pt`` file.
        model:
            Model to load state into.
        optimizer, scheduler, scaler:
            Optional objects to restore state into.
        device:
            Map tensors to this device.
        restore_rng:
            Whether to restore the saved RNG state (default: True).

        Returns
        -------
        dict
            The full checkpoint payload.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if restore_rng and "rng_state" in checkpoint:
            try:
                _restore_rng(checkpoint["rng_state"])
            except Exception as exc:
                logger.warning("Could not restore RNG state: %s", exc)

        logger.info(
            "Loaded checkpoint: %s  (epoch=%d  step=%d  loss=%.4f)",
            checkpoint_path,
            checkpoint.get("epoch", 0),
            checkpoint.get("global_step", 0),
            checkpoint.get("loss", float("nan")),
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def checkpoint_exists(self) -> bool:
        """Return True if ``latest.pt`` exists."""
        return self.latest_checkpoint.exists()

    def list_checkpoints(self) -> list[Path]:
        """Return all ``.pt`` files in the checkpoint directory, sorted."""
        return sorted(self.checkpoint_dir.glob("*.pt"))
