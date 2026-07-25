import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_dim: int,
        dropout: float,
    ):
        super().__init__()

        # Gate Projection
        self.gate_proj = nn.Linear(
            embedding_dim,
            feed_forward_dim,
            bias=False,
        )

        # Value Projection
        self.up_proj = nn.Linear(
            embedding_dim,
            feed_forward_dim,
            bias=False,
        )

        # Output Projection
        self.down_proj = nn.Linear(
            feed_forward_dim,
            embedding_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gate = F.silu(
            self.gate_proj(x)
        )

        value = self.up_proj(x)

        x = gate * value

        x = self.down_proj(x)

        x = self.dropout(x)

        return x