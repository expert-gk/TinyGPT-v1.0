from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Tokenizer(ABC):

    @abstractmethod
    def build_vocab(self, text: str) -> None:
        """Build the vocabulary from text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back into text."""
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save tokenizer state."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Load tokenizer state."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass