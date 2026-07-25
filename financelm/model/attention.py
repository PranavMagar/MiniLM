import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from financelm.model.rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        max_seq_length: int,
    ):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Rotary Positional Embedding
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_length=max_seq_length,
        )

        # One projection for Q, K and V
        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=False,
        )

        # Output projection
        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        # Causal Mask
        mask = torch.tril(
            torch.ones(
                max_seq_length,
                max_seq_length,
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                max_seq_length,
                max_seq_length,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        # -------------------------------------------------
        # QKV Projection
        # -------------------------------------------------
        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        # -------------------------------------------------
        # Split into heads
        # Shape:
        # (batch, heads, seq_len, head_dim)
        # -------------------------------------------------
        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # -------------------------------------------------
        # Apply Rotary Positional Embeddings
        # Only to Query and Key
        # -------------------------------------------------
        q = self.rope(q)
        k = self.rope(k)

        # -------------------------------------------------
        # Attention Scores
        # -------------------------------------------------
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.mask[
                :,
                :,
                :sequence_length,
                :sequence_length,
            ] == 0,
            float("-inf"),
        )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention,
        )

        # -------------------------------------------------
        # Weighted Sum
        # -------------------------------------------------
        output = attention @ v

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.embedding_dim,
            )
        )

        # -------------------------------------------------
        # Final Projection
        # -------------------------------------------------
        output = self.out_proj(output)

        return output