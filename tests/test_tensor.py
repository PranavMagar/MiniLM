import torch


def test_tensor_ops():
    x = torch.tensor([1, 2, 3])
    assert list(x.shape) == [3]
    print(x)
    print(x.shape)
