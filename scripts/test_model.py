"""Smoke test: FinanceLM forward pass and parameter count."""
import torch
from financelm.model.config import ModelConfig
from financelm.model.model import FinanceLM


def main() -> None:
    config = ModelConfig()
    model  = FinanceLM(config)

    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    logits    = model(input_ids)

    print("Input  shape :", input_ids.shape)
    print("Output shape :", logits.shape)
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
