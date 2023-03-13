import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size
        self.dropout = nn.Dropout(dropout)
        self.max_len = maxlen

        pos_embedding = self.positional_encoding()
        self.register_buffer("pos_embedding", pos_embedding)

    def positional_encoding(self):
        den = torch.exp(
            -torch.arange(0, self.emb_size, 2) * math.log(1000) / self.emb_size
        )
        pos = torch.arange(0, self.max_len).reshape(self.max_len, 1)

        pos_embedding = torch.zeros((self.max_len, self.emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        return pos_embedding

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )
