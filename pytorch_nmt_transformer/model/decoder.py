import torch.nn as nn
from pytorch_nmt_transformer.model import (
    CausalSelfAttention,
    CrossAttention,
    FeedForward,
    PositionalEncoding,
    TokenEmedding,
)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, device, dropout_rate=0.1) -> None:
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, embed_dim=d_model, dropout=dropout_rate, device=device
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, embed_dim=d_model, dropout=dropout_rate, device=device
        )
        self.ffn = FeedForward(d_model=d_model, dff=dff, device=device)

    def forward(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self, num_layers, d_model, num_heads, dff, vocab_size, device, dropout_rate=0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embedding = TokenEmedding(vocab_size=vocab_size, emb_size=d_model)
        self.pos_embedding = PositionalEncoding(emb_size=d_model, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                device=device,
            )
            for _ in range(num_heads)
        ]

    def forward(self, x, context):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, context)

        return x
