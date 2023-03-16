from argparse import ArgumentParser
from functools import partial
import logging
import pickle

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim import Adam

from pytorch_nmt_transformer import config
from pytorch_nmt_transformer.data import DataReader, Preprocessor
from pytorch_nmt_transformer.helper import collate_fn
from pytorch_nmt_transformer.model import SelfImplementedTransformer as Transformer
from pytorch_nmt_transformer.train import Trainer


logging.basicConfig(level=logging.INFO)


def load_vocab(file_path: str):
    with open(file_path, "rb") as fout:
        model = pickle.load(fout)
    return model


def run(args):
    logger = logging.getLogger("nmt_transformer")
    logger.info(f"Read Data from {args.dataset_path}")
    dr = DataReader(dataset_path=args.dataset_path)

    logger.info("Build vocab")
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
    pad_index = special_symbols.index("<pad>")
    preprocessor = Preprocessor(special_symbols=special_symbols)
    # preprocessor.build_vocab(dataset=dr)

    # with open(
    #     "./pytorch_nmt_transformer/saved_models/text_transformer.pickle", "wb"
    # ) as text_transform_pickle:
    #     pickle.dump(
    #         preprocessor.vocab_transforms,
    #         text_transform_pickle,
    #         protocol=pickle.HIGHEST_PROTOCOL,
    #     )

    vocab_transform = load_vocab(
        file_path="./pytorch_nmt_transformer/saved_models/text_transformer.pickle"
    )
    preprocessor.vocab_transforms = vocab_transform

    logger.info("Create collate function for training")
    text_transforms = preprocessor.get_all_transforms()
    collate_fn_partial = partial(
        collate_fn,
        text_transforms=text_transforms,
        pad_index=pad_index,
    )

    src_vocab_size = len(preprocessor.vocab_transforms[config.SOURCE_LANG])
    target_vocab_size = len(preprocessor.vocab_transforms[config.TARGET_LANG])

    logger.info("Create Model")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = Transformer(
        num_layers=config.NUM_LAYERS,
        d_model=config.EMB_SIZE,
        num_heads=config.NHEAD,
        input_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        dff=config.FFN_HID_DIM,
        device=DEVICE,
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer.to(device=DEVICE)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train_dataset, val_dataset = random_split(dataset=dr, lengths=[0.8, 0.2])

    logger.info("Train Model")
    trainer = Trainer(
        model=transformer,
        loss_fn=loss_fn,
        optimizer=optimizer,
        collate_fn=collate_fn_partial,
        device=DEVICE,
        pad_index=pad_index,
    )
    trainer.train(
        num_epochs=config.NUM_EPOCHS,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    logger.info("Saving Model")
    torch.save(
        transformer.state_dict(), "./pytorch_nmt_transformer/saved_models/model.ph"
    )

    logger.info("Models Saved")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path", "-d", required=True, help="Provide dataset text file path"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
