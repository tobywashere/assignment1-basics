import torch.nn as nn
import torch
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k isn't even"
        i = torch.arange(max_seq_len, device=device)[:, None]
        k = torch.arange(1, d_k//2+1, device=device)
        thetas = i / theta**((2*k-2)/d_k)
        cos = torch.cos(thetas)
        sin = torch.sin(thetas)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        paired = rearrange(x, "... (pair two) -> ... pair two", two=2)
        evens = paired[..., 0]
        odds = paired[..., 1]
        rotated_evens = evens * cos_pos - odds * sin_pos
        rotated_odds = evens * sin_pos + odds * cos_pos
        rotated = torch.stack([rotated_evens, rotated_odds], dim=-1)
        return rearrange(rotated, "... pair two -> ... (pair two)")
        

if __name__ == '__main__':
    theta = 100
    d_k = 128
    max_seq_len = 2048
    in_query_or_key = rearrange(torch.arange(max_seq_len * d_k), "(max_seq_len d_k) -> max_seq_len d_k", d_k=d_k)
    token_positions = torch.arange(max_seq_len)
    embedding = RotaryPositionalEmbedding(theta, d_k, max_seq_len)

    for _ in range(10):
        perm = torch.randperm(token_positions.shape[0])
        shuffled_token_positions = token_positions[perm]
        rotated = embedding(in_query_or_key, shuffled_token_positions)
        #print(f"tobyhuang debug: rotated={rotated}")
        assert(rotated.shape == (max_seq_len, d_k))        
