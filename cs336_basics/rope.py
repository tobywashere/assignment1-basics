import torch.nn as nn
import torch
from math import cos, sin
from einops import einsum

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        thetas = [[i / theta**((2*k-2)/d_k) for k in range(1, d_k//2 + 1)] for i in range(max_seq_len)]
        self.register_buffer("R", torch.zeros(max_seq_len, d_k, d_k, device=device), persistent=False)
        for i in range(max_seq_len):
            for k in range(d_k//2):
                self.R[i][2*k][2*k] = cos(thetas[i][k])
                self.R[i][2*k][2*k+1] = -sin(thetas[i][k])
                self.R[i][2*k+1][2*k] = sin(thetas[i][k])
                self.R[i][2*k+1][2*k+1] = cos(thetas[i][k])
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotations = self.R[token_positions]
        return einsum(rotations, x, "seq_len col_out row_in, ... seq_len row_in -> ... seq_len col_out")
        

if __name__ == '__main__':
    theta = 100
    d_k = 64
    max_seq_len = 12
    embedding = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    in_query_or_key = torch.ones(max_seq_len, d_k)
    token_positions = torch.arange(0, max_seq_len)
    perm = torch.randperm(token_positions.shape[0])
    shuffled_token_positions = token_positions[perm]
    rotated = embedding(in_query_or_key, shuffled_token_positions)
    print(f"tobyhuang debug: rotated={rotated}")
    assert(rotated.shape == (max_seq_len, d_k))
