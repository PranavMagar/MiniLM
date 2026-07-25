import torch

from financelm.model.embedding import TokenEmbedding


def main():
    embedding = TokenEmbedding(
        vocab_size=8192,
        embedding_dim=128,
    )

    input_ids = torch.randint(0, 8192, (4, 8))

    output = embedding(input_ids)

    print("Input shape :", input_ids.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()