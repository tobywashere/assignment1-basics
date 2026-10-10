import torch.nn as nn
import torch

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rope import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2])
        y = x + self.attn(self.ln1(x), self.rope, token_positions)
        return y + self.ffn(self.ln2(y))

if __name__ == '__main__':
    m = TransformerBlock(64, 4, 256, 10000, 16)
    input = torch.randn(300, 12, 64)
    output = m(input)
    assert output.shape == (300, 12, 64)
