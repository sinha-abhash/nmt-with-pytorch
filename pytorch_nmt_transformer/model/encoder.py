from torch import Tensor
import torch.nn as nn

from pytorch_nmt_transformer.model import (
    GlobalSelfAttention,
    FeedForward,
    PositionalEncoding,
    TokenEmedding,
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, device, dropout_rate=0.1) -> None:
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, embed_dim=d_model, dropout=dropout_rate, device=device
        )
        self.ffn = FeedForward(
            d_model=d_model, dff=dff, dropout_rate=dropout_rate, device=device
        )

    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads, dff, vocab_size, device, dropout_rate=0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embedding = TokenEmedding(vocab_size=vocab_size, emb_size=d_model)
        self.pos_embedding = PositionalEncoding(emb_size=d_model, dropout=dropout_rate)

        self.encoder_layers = [
            EncoderLayer(
                d_model=self.d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                device=device,
            )
            for _ in range(self.num_layers)
        ]

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

        return x
