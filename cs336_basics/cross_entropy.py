import torch
from jaxtyping import Float, Int
from cs336_basics.softmax import log_softmax

def cross_entropy(inputs: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, ""]:
    """
    Returns the mean loss over all batch dimensions
    """
    log_probabilities = log_softmax(inputs, dim=-1)
    selected = log_probabilities.gather(dim=-1, index=targets[..., None])
    return -selected.mean()
