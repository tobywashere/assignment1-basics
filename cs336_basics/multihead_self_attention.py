from einops import einsum, rearrange
import torch.nn as nn
import torch

from cs336_basics.scaled_dot_product_attention import attention
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.linear import Linear

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = self.d_v = d_model // num_heads
        self.q_proj = Linear(num_heads * self.d_k, d_model)
        self.k_proj = Linear(num_heads * self.d_k, d_model)
        self.v_proj = Linear(num_heads * self.d_v, d_model)
        self.output_proj = Linear(d_model, num_heads * self.d_v)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        proj = einsum(proj, x, "h3d_k d_model, ... seq_len d_model -> ... seq_len h3d_k")
        q_proj = proj.narrow(dim=-1, start=0, length=self.num_heads*self.d_k)
        k_proj = proj.narrow(dim=-1, start=self.num_heads*self.d_k, length=self.num_heads*self.d_k)
        v_proj = proj.narrow(dim=-1, start=2*self.num_heads*self.d_k, length=self.num_heads*self.d_v)
        q_proj = rearrange(q_proj, "... seq_len (h d_k) -> h ... seq_len d_k", h=self.num_heads)
        k_proj = rearrange(k_proj, "... seq_len (h d_k) -> h ... seq_len d_k", h=self.num_heads)
        v_proj = rearrange(v_proj, "... seq_len (h d_v) -> h ... seq_len d_v", h=self.num_heads)
        if rope is not None:
            q_proj = rope(q_proj, token_positions)
            k_proj = rope(k_proj, token_positions)
        i = torch.arange(seq_len)[:, None]
        j = torch.arange(seq_len)[None, :]
        mask = j <= i
        multihead = attention(q_proj, k_proj, v_proj, mask)
        multihead = rearrange(multihead, "h ... seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(multihead)        
