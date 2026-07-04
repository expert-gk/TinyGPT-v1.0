from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import load_config
from src.tokenizers.tokenizer_factory import TokenizerFactory
from src.tokenizers.trainer import TokenizerTrainer
from src.tokenizers.word_tokenizer import WordTokenizer


def train_tokenizer() -> None:

    config = load_config("configs/default.yaml")

    # tokenizer = TokenizerFactory.create(config)

    tokenizer = WordTokenizer.load(
        "datasets/tokenizer.json"
    )

    trainer = TokenizerTrainer(tokenizer)

    output_file = Path("datasets/tokenizer.json")

    trainer.train(
        input_file=config.dataset.train_file,
        output_file=str(output_file),
    )

    print(f"Tokenizer saved to {output_file}")


def run() -> None:

    parser = argparse.ArgumentParser(
        prog="TinyGPT"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    tokenizer = subparsers.add_parser("tokenizer")

    tokenizer.add_argument(
        "action",
        choices=["train"],
    )

    subparsers.add_parser("train")

    subparsers.add_parser("generate")

    args = parser.parse_args()

    if args.command == "tokenizer":

        if args.action == "train":
            train_tokenizer()

        return

    print(f"{args.command} not implemented yet.")