import math

from torch import Tensor
import torch.nn as nn


class TokenEmedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super(TokenEmedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
