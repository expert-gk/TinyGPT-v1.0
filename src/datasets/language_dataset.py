from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.tokenizers.base import Tokenizer
from src.config.schema import GPTConfig


class LanguageDataset(Dataset):

    def __init__(
        self,
        config: GPTConfig,
        tokenizer: Tokenizer,
        file_path: str,
    ) -> None:

        self.context_length = config.model.context_length

        text = Path(file_path).read_text(
            encoding="utf-8"
        )

        # tokenizer.build_vocab(text)

        self.tokenizer = tokenizer

        self.tokens = tokenizer.encode(text)

    def __len__(self) -> int:

        return len(self.tokens) - self.context_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.tokens[
            index:index + self.context_length
        ]

        y = self.tokens[
            index + 1:index + self.context_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )