from argparse import ArgumentParser
import logging
from pathlib import Path
import pickle

import torch

from pytorch_nmt_transformer.data import Preprocessor
from pytorch_nmt_transformer.model import Seq2SeqTransformer
from pytorch_nmt_transformer import config
from pytorch_nmt_transformer.translator import Translator


logging.basicConfig(level=logging.INFO)


class NotCorrectModelPathException(Exception):
    pass


def check_if_model_exists(path: Path) -> bool:
    if not path.exists:
        raise FileNotFoundError(f"Provided path does not exists: {str(path)}")

    all_files = list(path.iterdir())
    model_files = [f for f in all_files if f.suffix == ".ph"]
    vocab_file = [f for f in all_files if f.suffix == ".pickle"]

    if len(model_files) == 0:
        raise NotCorrectModelPathException("Provided path does not have model files.")
    if len(vocab_file) == 0:
        raise NotCorrectModelPathException(
            "Provided path does not have trained vocab file."
        )
    return True


def load_vocab(file_path: str):
    with open(file_path, "rb") as fout:
        model = pickle.load(fout)
    return model


def load_model(
    file_path: str, src_vocab_size: int, target_vocab_size: int
) -> Seq2SeqTransformer:
    model_class = Seq2SeqTransformer(
        num_encoders=config.NUM_LAYERS,
        num_decoders=config.NUM_LAYERS,
        emb_size=config.EMB_SIZE,
        num_head=config.NHEAD,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        ff_dim=config.FFN_HID_DIM,
    )

    model_class.load_state_dict(torch.load(file_path))
    return model_class


def infer(args):
    logger = logging.getLogger("infer")
    check_if_model_exists(Path(args.model_path))

    logger.info("Load vocab transform")
    vocab_transform = load_vocab(file_path=f"{args.model_path}/text_transformer.pickle")

    logger.info("Prepare text transform")
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
    preprocessor = Preprocessor(special_symbols)
    preprocessor.vocab_transforms = vocab_transform
    text_transforms = preprocessor.get_all_transforms()

    src_vocab_size = len(preprocessor.vocab_transforms[config.SOURCE_LANG])
    target_vocab_size = len(preprocessor.vocab_transforms[config.TARGET_LANG])

    logger.info("Load the model")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        file_path=f"{args.model_path}/model.ph",
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
    )
    model = model.to(DEVICE)

    logger.info("Translate")
    translator = Translator(
        trained_model=model,
        text_transform=text_transforms,
        vocab_transform=vocab_transform,
        bos_index=special_symbols.index("<bos>"),
        eos_index=special_symbols.index("<eos>"),
        device=DEVICE,
    )

    output_text = translator.translate(args.input_text)
    logger.info(f"Translation of {args.input_text} is: \n{output_text}")


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--model_path", "-m", required=True, help="Provide trained model path"
    )
    arg_parser.add_argument(
        "--input_text", "-i", required=True, help="Provide input text for translation"
    )

    args = arg_parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
