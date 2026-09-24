import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies softmax to the `dim` dimension of the input tensor
    """
    max_elem = x.amax(dim=dim, keepdim=True)
    expo = torch.exp(x - max_elem)
    sums = expo.sum(dim=dim, keepdim=True)
    return expo / sums

if __name__ == '__main__':
    # A tensor of fully-masked rows produce NaN and poisons the output, which shouldn't occur here
    mask = torch.tensor([[-float('inf'), -float('inf')]])
    result = softmax(mask, 1)
    print(f"tobyhuang debug: fully masked row produces NaNs: {result}")
    # sanity check: partially-masked row softmaxes correctly
    mask = torch.tensor([[1, -float('inf'), -float('inf')]])
    result = softmax(mask, 1)
    print(f"tobyhuang debug: partially masked row softmaxes correctly: {result}")
