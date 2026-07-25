"""Smoke test: TransformerBlock forward pass."""
import torch
from financelm.model.config import ModelConfig
from financelm.model.embedding import TokenEmbedding
from financelm.model.transformer_block import TransformerBlock


def main() -> None:
    config    = ModelConfig()
    embedding = TokenEmbedding(config.vocab_size, config.embedding_dim)
    block     = TransformerBlock(config)

    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    x = embedding(input_ids)
    print("Before block:", x.shape)
    x = block(x)
    print("After block :", x.shape)


if __name__ == "__main__":
    main()
