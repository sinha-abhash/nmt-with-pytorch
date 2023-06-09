import torch.nn as nn
from torch import Tensor

from pytorch_nmt_transformer.model import TokenEmedding, PositionalEncoding


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoders: int,
        num_decoders: int,
        emb_size: int,
        num_head: int,
        src_vocab_size: int,
        target_vocab_size: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=num_head,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_decoders,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )

        self.generator = nn.Linear(emb_size, target_vocab_size)
        self.src_tok_emb = TokenEmedding(src_vocab_size, emb_size)
        self.target_tok_emb = TokenEmedding(target_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(
            emb_size=emb_size, dropout=dropout
        )

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Tensor,
        target_mask: Tensor,
        src_padding_mask: Tensor,
        target_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        target_emb = self.positional_encoding(self.target_tok_emb(target))

        outs = self.transformer(
            src_emb,
            target_emb,
            src_mask,
            target_mask,
            None,
            src_padding_mask,
            target_padding_mask,
            memory_key_padding_mask,
        )

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.target_tok_emb(target)), memory, target_mask
        )
