import torch.nn as nn
import torch
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i = torch.arange(0, max_seq_len)
        i = rearrange(i, "... -> ... 1")
        k = torch.arange(1, d_k//2+1)
        thetas = i / theta**((2*k-2)/d_k)
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        evens = x[..., 0::2]
        odds = x[..., 1::2]
        result = torch.stack([evens * cos - odds * sin, evens * sin + odds * cos], dim=-1)
        return rearrange(result, "... pair two -> ... (pair two)")
        

if __name__ == '__main__':
    theta = 100
    d_k = 128
    max_seq_len = 2048
    embedding = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    in_query_or_key = torch.ones(max_seq_len, d_k)
    token_positions = torch.arange(0, max_seq_len)
    perm = torch.randperm(token_positions.shape[0])
    shuffled_token_positions = token_positions[perm]
    rotated = embedding(in_query_or_key, shuffled_token_positions)
    print(f"tobyhuang debug: rotated={rotated}")
    assert(rotated.shape == (max_seq_len, d_k))
