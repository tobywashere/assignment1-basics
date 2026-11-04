import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies softmax to the `dim` dimension of the input tensor
    """
    return torch.exp(log_softmax(x, dim))

def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Returns stable log-probabilities along dim
    A tensor of fully-masked rows produce NaN and poisons the output.
    The caller is responsible for ensuring this case doesn't happen.
    """
    max_elem = x.amax(dim=dim, keepdim=True)
    shifted = x - max_elem
    expo = torch.exp(shifted)
    sums = expo.sum(dim=dim, keepdim=True)
    return shifted - torch.log(sums)

if __name__ == '__main__':
    import numpy
    # Test fully-masked rows produce NaN
    mask = torch.tensor([[-float('inf'), -float('inf')]])
    result = softmax(mask, 1)
    assert result.isnan().all()
    print(f"tobyhuang debug: fully masked row produces NaNs: {result}")
    # sanity check: partially-masked row softmaxes correctly
    mask = torch.tensor([[1, -float('inf'), -float('inf')]])
    result = softmax(mask, 1)
    numpy.testing.assert_allclose(
        result,
        torch.tensor([[1, 0, 0]]),
        atol=1e-4,
    )
    print(f"tobyhuang debug: partially masked row softmaxes correctly: {result}")
