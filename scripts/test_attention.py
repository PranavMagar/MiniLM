"""Smoke test: MultiHeadSelfAttention forward pass."""
import torch
from financelm.model.config import ModelConfig
from financelm.model.embedding import TokenEmbedding
from financelm.model.attention import MultiHeadSelfAttention


def main() -> None:
    config    = ModelConfig()
    embedding = TokenEmbedding(config.vocab_size, config.embedding_dim)
    attention = MultiHeadSelfAttention(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_seq_length=config.max_seq_length,
    )

    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    x = embedding(input_ids)
    print("Before attention:", x.shape)
    x = attention(x)
    print("After attention :", x.shape)


if __name__ == "__main__":
    main()
