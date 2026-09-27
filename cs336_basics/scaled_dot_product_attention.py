from cs336_basics.softmax import softmax

from einops import einsum
from jaxtyping import Bool, Float
from math import sqrt
import torch

def attention(Q: Float[torch.Tensor, " ... queries d_k"],
              K: Float[torch.Tensor, " ... keys d_k"],
              V: Float[torch.Tensor, " ... keys d_v"],
              mask: Bool[torch.Tensor, " ... queries keys"] | None = None) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)
    masked_scores = pre_softmax
    if mask is not None:
        masked_scores = torch.where(mask, pre_softmax, -float('inf'))
    post_softmax = softmax(masked_scores, dim=-1)
    return einsum(post_softmax, V, "... queries keys, ... keys d_v -> ... queries d_v")

if __name__ == '__main__':
    d_k = 64
    d_v = 64
    queries = 12
    keys = 16
    Q = torch.randn(queries, d_k, requires_grad=True)
    K = torch.randn(keys, d_k, requires_grad=True)
    V = torch.randn(keys, d_v, requires_grad=True)
    result = attention(Q, K, V)
    mask = torch.le(torch.randn(queries, keys), 0)
    result = attention(Q, K, V, mask)
    result.sum().backward()
    print(f"tobyhuang debug: result={result}, grad={result.grad}")
