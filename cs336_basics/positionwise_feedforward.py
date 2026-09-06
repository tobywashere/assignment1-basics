import torch
import torch.nn as nn
from math import sqrt
from einops import einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # d_ff should equal d_model * 8 / 3 rounded to the nearest 64 for hardware efficiency
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        sig = sqrt(2/(d_model + d_ff))
        nn.init.trunc_normal_(self.W1, std=sig, a=-3*sig, b=3*sig)
        nn.init.trunc_normal_(self.W2, std=sig, a=-3*sig, b=3*sig)
        nn.init.trunc_normal_(self.W3, std=sig, a=-3*sig, b=3*sig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu = w1_x * torch.sigmoid(w1_x)
        w3_x = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        rhs = silu * w3_x
        return einsum(self.W2, rhs, "d_model d_ff, ... d_ff -> ... d_model")
        

if __name__ == '__main__':
    m = SwiGLU(100, 256)
    input = torch.randn(300, 200, 100)
    output = m(input)
    assert(output.shape == (300, 200, 100))
