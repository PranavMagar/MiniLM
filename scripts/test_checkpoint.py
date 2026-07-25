"""Smoke test: CheckpointManager save and load round-trip."""
import torch
from torch.optim import AdamW
from financelm.model.config import ModelConfig
from financelm.model.model import FinanceLM
from financelm.training.checkpoint import CheckpointManager


def main() -> None:
    config    = ModelConfig()
    model     = FinanceLM(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    manager   = CheckpointManager()

    path = manager.save_latest(
        model=model, optimizer=optimizer,
        epoch=1, loss=5.0, config=config, global_step=100,
    )
    print(f"Saved  : {path}")

    ckpt = manager.load(checkpoint_path=path, model=model)
    print(f"Loaded : epoch={ckpt['epoch']}  step={ckpt['global_step']}  loss={ckpt['loss']}")
    print("Checkpoint test passed.")


if __name__ == "__main__":
    main()
