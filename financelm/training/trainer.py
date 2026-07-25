"""
financelm.training.trainer
===========================
Training and validation loop for FinanceLM.

Supports:
- Mixed precision (AMP) on CUDA with correct unscale → clip → step order
- Gradient clipping
- Per-step loss accumulation
- Optional validation loop
- tqdm progress bars
- Global step tracking
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for a decoder-only language model.

    Parameters
    ----------
    model:
        The language model to train.
    optimizer:
        PyTorch optimizer.
    scheduler:
        Learning-rate scheduler (stepped once per epoch).
    device:
        Target device.
    gradient_clip:
        Max-norm for gradient clipping (default: 1.0).
    global_step:
        Starting global step (for resume; default: 0).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        gradient_clip: float = 1.0,
        global_step: int = 0,
    ) -> None:
        self.model         = model
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.gradient_clip = gradient_clip
        self.global_step   = global_step

        self.use_amp = device.type == "cuda"
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> float:
        """
        Run one training epoch.

        Parameters
        ----------
        dataloader:
            Training data loader.
        epoch:
            Current epoch number (for tqdm display only).

        Returns
        -------
        float
            Mean cross-entropy loss over the epoch.
        """
        self.model.train()
        total_loss  = 0.0
        n_batches   = len(dataloader)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:02d} [train]",
            leave=False,
            dynamic_ncols=True,
        )

        for input_ids, target_ids in pbar:
            input_ids  = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(input_ids)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )

            self.scaler.scale(loss).backward()

            # Unscale before clipping — required for correct gradient norms
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss       += loss.item()
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", step=self.global_step)

        self.scheduler.step()
        return total_loss / n_batches

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> float:
        """
        Run one validation epoch.

        Parameters
        ----------
        dataloader:
            Validation data loader.
        epoch:
            Current epoch number (for tqdm display only).

        Returns
        -------
        float
            Mean cross-entropy loss over the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches  = len(dataloader)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:02d} [val]",
            leave=False,
            dynamic_ncols=True,
        )

        for input_ids, target_ids in pbar:
            input_ids  = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(input_ids)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                )

            total_loss += loss.item()
            pbar.set_postfix(val_loss=f"{loss.item():.4f}")

        return total_loss / n_batches
