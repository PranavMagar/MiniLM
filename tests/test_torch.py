import torch


def test_tensor_creation():
    x = torch.tensor([1, 2, 3])
    assert x.shape == (3,)
    print("PyTorch version:", torch.__version__)
    print("Tensor:", x)
