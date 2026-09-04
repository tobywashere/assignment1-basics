import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sig = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.W, std=sig, a=-3*sig, b=3*sig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")

if __name__ == '__main__':
    m = Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())
