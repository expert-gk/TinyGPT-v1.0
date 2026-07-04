from src.config.schema import GPTConfig
from src.tokenizers.base import Tokenizer
from src.tokenizers.word_tokenizer import WordTokenizer


class TokenizerFactory:

    @staticmethod
    def create(config: GPTConfig) -> Tokenizer:

        tokenizer_type = config.tokenizer.type.lower()

        if tokenizer_type == "word":
            return WordTokenizer()

        raise ValueError(
            f"Unsupported tokenizer: {tokenizer_type}"
        )