from einops import einsum, rearrange
import torch.nn as nn
import torch

from cs336_basics.scaled_dot_product_attention import attention
from cs336_basics.rope import RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        d_k = d_v = d_model // num_heads
        self.Q = nn.Parameter(torch.empty(num_heads * d_k, d_model))
        self.K = nn.Parameter(torch.empty(num_heads * d_k, d_model))
        self.V = nn.Parameter(torch.empty(num_heads * d_v, d_model))
        self.O = nn.Parameter(torch.empty(d_model, num_heads * d_v))

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        Qx = einsum(self.Q, x, "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k")
        Kx = einsum(self.K, x, "hd_k d_model, ... seq_len d_model -> ... seq_len hd_k")
        Vx = einsum(self.V, x, "hd_v d_model, ... seq_len d_model -> ... seq_len hd_v")
        Qx = rearrange(Qx, "... seq_len (h d_k) -> h ... seq_len d_k", h=self.num_heads)
        Kx = rearrange(Kx, "... seq_len (h d_k) -> h ... seq_len d_k", h=self.num_heads)
        Vx = rearrange(Vx, "... seq_len (h d_v) -> h ... seq_len d_v", h=self.num_heads)
        if rope is not None:
            Qx = rope(Qx, token_positions)
            Kx = rope(Kx, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        multihead = torch.cat([attention(Qx[i], Kx[i], Vx[i], mask) for i in range(self.num_heads)], dim=-1)
        return einsum(self.O, multihead, "d_model hd_v, ... seq_len hd_v -> ... seq_len d_model")
        
