from cs336_basics.softmax import softmax

from einops import einsum
from jaxtyping import Bool, Float
from math import sqrt
from torch import randn, Tensor, where


def attention(Q: Float[Tensor, " ... queries d_k"],
              K: Float[Tensor, " ... keys d_k"],
              V: Float[Tensor, " ... keys d_v"],
              mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    pre_softmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)
    if mask is not None:
        masked_scores = where(mask, pre_softmax, -float('inf'))
    else:
        masked_scores = pre_softmax
    post_softmax = softmax(masked_scores, dim=-1)
    return einsum(post_softmax, V, "... queries keys, ... keys d_v -> ... queries d_v")

if __name__ == '__main__':
    d_k = 64
    d_v = 64
    queries = 12
    keys = 16
    Q = randn(queries, d_k, requires_grad=True)
    K = randn(keys, d_k, requires_grad=True)
    V = randn(keys, d_v, requires_grad=True)
    result = attention(Q, K, V)
    print(result)
