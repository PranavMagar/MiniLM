import torch
import torch.nn as nn

from financelm.model.embedding import TokenEmbedding
from financelm.model.transformer_block import TransformerBlock
from financelm.model.rms_norm import RMSNorm


class FinanceLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Token Embedding
        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.embedding_dim,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        # Final RMSNorm
        self.final_norm = RMSNorm(config.embedding_dim)

        # Language Modeling Head
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # -------------------------------------------------
        # Weight Tying
        # Share token embedding and output projection weights
        # -------------------------------------------------
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(self, input_ids):

        x = self.token_embedding(input_ids)

        # RoPE supplies positional information inside attention.
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits