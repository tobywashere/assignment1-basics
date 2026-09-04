import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]

if __name__ == '__main__':
    embedding = Embedding(10, 3)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(embedding(input))
    assert(embedding(input).shape == (2, 4, 3))
