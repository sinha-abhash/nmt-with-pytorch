import logging
from typing import Tuple
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pytorch_nmt_transformer import config
from pytorch_nmt_transformer.model import Seq2SeqTransformer


def collate_fn(batch: list, text_transforms: dict, pad_index: int) -> Tuple[list, list]:
    src_batch, target_batch = [], []
    for src, target in batch:
        src_batch.append(text_transforms[config.SOURCE_LANG](src))
        target_batch.append(text_transforms[config.TARGET_LANG](target))

    src_batch = pad_sequence(src_batch, padding_value=pad_index)
    target_batch = pad_sequence(target_batch, padding_value=pad_index)
    return src_batch, target_batch


def generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(
    src: Tensor, target: Tensor, pad_index: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    src_seq_len = src.shape[0]
    target_seq_len = target.shape[0]

    target_mask: Tensor = generate_square_subsequent_mask(target_seq_len, device=device)
    src_mask: Tensor = torch.zeros((src_seq_len, src_seq_len), device=device).type(
        torch.bool
    )

    src_padding_mask: Tensor = (src == pad_index).transpose(0, 1)
    target_padding_mask: Tensor = (target == pad_index).transpose(0, 1)

    return src_mask, target_mask, src_padding_mask, target_padding_mask


def greedy_decode(
    model: Seq2SeqTransformer,
    src: Tensor,
    src_mask: Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: torch.device,
):
    logger = logging.getLogger("greedy_decode")
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for _ in range(max_len - 1):
        memory = memory.to(device)
        target_mask = (
            generate_square_subsequent_mask(ys.size(0), device=device).type(torch.bool)
        ).to(device)

        out = model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    return ys
