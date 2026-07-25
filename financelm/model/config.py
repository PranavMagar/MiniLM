from dataclasses import dataclass


@dataclass
class ModelConfig:

    # Tokenizer
    vocab_size: int = 8192

    # Model Size (~10M Parameters)
    embedding_dim: int = 256
    num_heads: int = 4
    num_layers: int = 8
    feed_forward_dim: int = 1024

    # Context
    max_seq_length: int = 256

    # Training
    dropout: float = 0.1
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 10

    # Checkpoints
    checkpoint_interval: int = 10