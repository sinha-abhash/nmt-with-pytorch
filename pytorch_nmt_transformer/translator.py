import torch
import torch.nn as nn

from pytorch_nmt_transformer.helper import greedy_decode
from pytorch_nmt_transformer import config


class Translator:
    def __init__(
        self,
        trained_model: nn.Module,
        text_transform: dict,
        vocab_transform: dict,
        bos_index: int,
        eos_index: int,
        device: torch.device,
    ) -> None:
        self.model = trained_model
        self.text_transform = text_transform
        self.vocab_transform = vocab_transform
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.device = device

    def translate(self, src_text: str) -> str:
        self.model.eval()

        src = self.text_transform[config.SOURCE_LANG](src_text).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        target_tokens = greedy_decode(
            model=self.model,
            src=src,
            src_mask=src_mask,
            max_len=num_tokens + 5,
            start_symbol=self.bos_index,
            end_symbol=self.eos_index,
            device=self.device,
        ).flatten()

        return (
            " ".join(
                self.vocab_transform[config.TARGET_LANG].lookup_tokens(
                    list(target_tokens.cpu().numpy())
                )
            )
            .replace("<bos>", "")
            .replace("<eos>", "")
        )
