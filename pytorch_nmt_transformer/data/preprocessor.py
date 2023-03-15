import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

from typing import List

from pytorch_nmt_transformer import config


class Preprocessor:
    def __init__(self, special_symbols: List[str]) -> None:
        self.special_symbols = special_symbols

        self.tokenizers = Preprocessor.get_tokenizers()
        self.vocab_transforms = {}
        self.text_transforms = {}

        self.bos_index = self.special_symbols.index("<bos>")
        self.eos_index = self.special_symbols.index("<eos>")

    @staticmethod
    def get_tokenizers() -> dict:
        return {
            config.SOURCE_LANG: get_tokenizer("spacy", language="de_core_news_sm"),
            config.TARGET_LANG: get_tokenizer("spacy", language="en_core_web_sm"),
        }

    def yield_tokens(self, dataset: Dataset, language: str):
        language_index = {config.SOURCE_LANG: 0, config.TARGET_LANG: 1}
        for data in dataset:
            yield self.tokenizers[language](data[language_index[language]])

    def build_vocab(self, dataset: Dataset):
        for lang in [config.SOURCE_LANG, config.TARGET_LANG]:
            self.vocab_transforms[lang] = build_vocab_from_iterator(
                self.yield_tokens(dataset, lang),
                min_freq=1,
                specials=self.special_symbols,
                special_first=True,
            )

    def get_sequential_transformations(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)

            return txt_input

        return func

    def tensor_transform(self, token_ids: List[int]):
        return torch.cat(
            (
                torch.tensor([self.bos_index]),
                torch.tensor(token_ids),
                torch.tensor([self.eos_index]),
            )
        )

    def get_all_transforms(self):
        for lang in [config.SOURCE_LANG, config.TARGET_LANG]:
            self.text_transforms[lang] = self.get_sequential_transformations(
                self.tokenizers[lang],
                self.vocab_transforms[lang],
                self.tensor_transform,
            )
        return self.text_transforms
