import torch.nn as nn
import torch

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.positionwise_feedforward import SwiGLU
from cs336_basics.rope import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, rope: RotaryPositionalEmbedding | None = None, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), rope, token_positions)
        return y + self.ffn(self.ln2(y))

if __name__ == '__main__':
    m = TransformerBlock(100, 4, 256)
    input = torch.randn(300, 200, 100)
    output = m(input)
    assert output.shape == (300, 200, 100)
