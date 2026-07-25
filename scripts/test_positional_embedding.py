"""Smoke test: TokenEmbedding + PositionalEmbedding forward pass."""
import torch
from financelm.model.config import ModelConfig
from financelm.model.embedding import TokenEmbedding
from financelm.model.positional_embedding import PositionalEmbedding


def main() -> None:
    config    = ModelConfig()
    tok_emb   = TokenEmbedding(config.vocab_size, config.embedding_dim)
    pos_emb   = PositionalEmbedding(config.max_seq_length, config.embedding_dim)

    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    x = tok_emb(input_ids)
    print("After token embedding    :", x.shape)
    x = pos_emb(x)
    print("After positional embedding:", x.shape)


if __name__ == "__main__":
    main()
