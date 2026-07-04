from __future__ import annotations

from pathlib import Path

from src.tokenizers.base import Tokenizer


class TokenizerTrainer:

    def __init__(
        self,
        tokenizer: Tokenizer,
    ) -> None:

        self.tokenizer = tokenizer

    def train(
        self,
        input_file: str,
        output_file: str,
    ) -> None:

        text = Path(
            input_file
        ).read_text(
            encoding="utf-8"
        )

        self.tokenizer.build_vocab(
            text
        )

        self.tokenizer.save(
            output_file
        )