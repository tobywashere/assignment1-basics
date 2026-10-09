import torch
import torch.nn as nn
from math import sqrt
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # d_ff should equal d_model * 8 / 3 rounded to the nearest 64 for hardware efficiency
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        rhs = silu * w3_x
        return self.w2(rhs)
        

if __name__ == '__main__':
    m = SwiGLU(100, 256)
    input = torch.randn(300, 200, 100)
    output = m(input)
    assert(output.shape == (300, 200, 100))
