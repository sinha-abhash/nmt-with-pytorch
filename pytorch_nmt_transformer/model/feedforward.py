import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, device, dropout_rate=0.1) -> None:
        super().__init__()
        self.seqential_ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.seqential_ffn.to(device=device)
        self.layer_norm.to(device=device)

    def forward(self, x):
        seq_output = self.seqential_ffn(x)
        x = x + seq_output
        x = self.layer_norm(x)

        return x
