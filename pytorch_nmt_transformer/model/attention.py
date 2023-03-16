import torch.nn as nn


class BaseAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.mha = nn.MultiheadAttention(**kwargs)
        self.layernorm = nn.LayerNorm(kwargs["embed_dim"])

        self.mha.to(kwargs["device"])
        self.layernorm.to(kwargs["device"])


class CrossAttention(BaseAttention):
    def forward(self, x, context):
        attn_output, attn_output_weights = self.mha(query=x, key=context, value=context)
        x = x + attn_output
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def forward(self, x):
        attn_output, attn_output_weights = self.mha(query=x, key=x, value=x)
        x = x + attn_output
        x = self.layernorm(x)

        return x


class CausalSelfAttention(BaseAttention):
    def forward(self, x):
        attn_output, attn_output_weights = self.mha(
            query=x, key=x, value=x, is_causal=True
        )
        x = x + attn_output
        x = self.layernorm(x)

        return x
