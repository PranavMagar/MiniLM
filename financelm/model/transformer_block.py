import torch.nn as nn

from financelm.model.rms_norm import RMSNorm
from financelm.model.attention import MultiHeadSelfAttention
from financelm.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = RMSNorm(config.embedding_dim)

        self.attention = MultiHeadSelfAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_length=config.max_seq_length,
        )

        self.norm2 = RMSNorm(config.embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=config.embedding_dim,
            feed_forward_dim=config.feed_forward_dim,
            dropout=config.dropout,
        )

    def forward(self, x):

        x = x + self.attention(self.norm1(x))

        x = x + self.feed_forward(self.norm2(x))

        return x