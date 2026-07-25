import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.
    """

    def __init__(self, max_seq_length: int, embedding_dim: int):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        input_embeddings shape:
            (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length, _ = input_embeddings.shape

        positions = torch.arange(
            sequence_length,
            device=input_embeddings.device,
        )

        positions = positions.unsqueeze(0).expand(batch_size, sequence_length)

        position_embeddings = self.position_embedding(positions)

        return input_embeddings + position_embeddings