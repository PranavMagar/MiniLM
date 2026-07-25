"""Smoke test: RMSNorm forward pass."""
import torch
from financelm.model.config import ModelConfig
from financelm.model.embedding import TokenEmbedding
from financelm.model.rms_norm import RMSNorm


def main() -> None:
    config    = ModelConfig()
    embedding = TokenEmbedding(config.vocab_size, config.embedding_dim)
    norm      = RMSNorm(config.embedding_dim)

    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    x = embedding(input_ids)
    print("Before RMSNorm:", x.shape)
    x = norm(x)
    print("After RMSNorm :", x.shape)


if __name__ == "__main__":
    main()
