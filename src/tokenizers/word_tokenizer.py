from __future__ import annotations

import json
import re
from pathlib import Path

from src.tokenizers.base import Tokenizer


class WordTokenizer(Tokenizer):
    """
    A simple word-level tokenizer.

    Converts text into integer token IDs and
    supports saving/loading the vocabulary.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self) -> None:

        self.word_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        self.id_to_word: dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }

    def build_vocab(self, text: str) -> None:

        words = re.findall(r"\b\w+\b", text.lower())

        for word in sorted(set(words)):

            if word not in self.word_to_id:

                index = len(self.word_to_id)

                self.word_to_id[word] = index

                self.id_to_word[index] = word

    def encode(self, text: str) -> list[int]:

        words = re.findall(r"\b\w+\b", text.lower())

        return [
            self.word_to_id.get(word, 1)
            for word in words
        ]

    def decode(self, tokens: list[int]) -> str:

        return " ".join(
            self.id_to_word.get(token, self.UNK_TOKEN)
            for token in tokens
        )

    @property
    def vocab_size(self) -> int:

        return len(self.word_to_id)

    def save(self, path: str | Path) -> None:

        path = Path(path)

        data = {
            "word_to_id": self.word_to_id
        }

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":

        path = Path(path)

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        tokenizer = cls()

        tokenizer.word_to_id = data["word_to_id"]

        tokenizer.id_to_word = {
            int(v): k
            for k, v in tokenizer.word_to_id.items()
        }

        return tokenizer
    
    @property
    def is_trained(self) -> bool:
        return self.vocab_size > 2