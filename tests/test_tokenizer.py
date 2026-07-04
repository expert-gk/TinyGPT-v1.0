from pathlib import Path

from src.config.settings import load_config
from src.tokenizers.tokenizer_factory import TokenizerFactory


def main() -> None:

    config = load_config("configs/default.yaml")

    tokenizer = TokenizerFactory.create(config)

    text = (
        Path("datasets/train.txt")
        .read_text(encoding="utf-8")
    )

    tokenizer.build_vocab(text)

    print("Vocabulary:", tokenizer.vocab_size)

    encoded = tokenizer.encode(
        "India is a country"
    )

    print(encoded)

    print(
        tokenizer.decode(encoded)
    )

    tokenizer.save("datasets/tokenizer.json")


if __name__ == "__main__":
    main()