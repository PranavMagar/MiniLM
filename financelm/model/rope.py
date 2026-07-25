import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)

    Applies rotary position embeddings to Query and Key tensors.

    Expected Input Shape:
        (batch_size, num_heads, seq_len, head_dim)
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_length: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for Rotary Embeddings."
            )

        self.head_dim = head_dim
        self.max_seq_length = max_seq_length

        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_seq_length,
            dtype=torch.float32,
        )

        freqs = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos",
            torch.cos(freqs),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            torch.sin(freqs),
            persistent=False,
        )

    def _rotate_half(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rotated = torch.stack(
            (-x_odd, x_even),
            dim=-1,
        )

        return rotated.flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        seq_len = x.size(-2)

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        x_rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        x = torch.stack(
            (
                x_rotated_even,
                x_rotated_odd,
            ),
            dim=-1,
        )

        return x.flatten(-2)