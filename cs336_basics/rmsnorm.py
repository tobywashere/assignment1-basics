from einops import einsum, rearrange
import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_a = einsum(x, x, "batch_size seq_len d_model, batch_size seq_len d_model -> batch_size seq_len")
        rms_a /= self.d_model
        rms_a += self.eps
        rms_a = torch.sqrt(rms_a)
        rms_a = rearrange(rms_a, "batch seq_len -> batch seq_len 1")
        result = x / rms_a * self.g

        return result.to(in_dtype)

if __name__ == '__main__':
    rms_norm = RMSNorm(4)
    input = torch.randn(2, 3, 4)
    output = rms_norm(input)
    print(output)
    assert(output.shape == (2, 3, 4))
