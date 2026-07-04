from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.config.settings import load_config
from src.datasets.language_dataset import LanguageDataset
from src.models.gpt import GPTModel
from src.tokenizers.word_tokenizer import WordTokenizer
from src.trainer.trainer import Trainer
from src.utils.device import get_device


def main() -> None:

    # --------------------------------------------------
    # Load configuration
    # --------------------------------------------------
    config = load_config("configs/default.yaml")

    device = get_device(config)

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load trained tokenizer
    # --------------------------------------------------
    tokenizer = WordTokenizer.load(
        "datasets/tokenizer.json"
    )

    print(f"Vocabulary Size: {tokenizer.vocab_size}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    train_dataset = LanguageDataset(
        config=config,
        tokenizer=tokenizer,
        file_path=config.dataset.train_file,
    )

    x, y = train_dataset[0]
    print(x)
    print(y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    print(f"Training Samples: {len(train_dataset)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.model.embedding_dimension,
        context_length=config.model.context_length,
        num_heads=config.model.number_of_heads,
        num_layers=config.model.number_of_layers,
        dropout=config.model.dropout,
    )

    print(
        f"Model Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
    )

    # --------------------------------------------------
    # Training Loop
    # --------------------------------------------------
    print()

    print("Starting Training...\n")

    for epoch in range(config.training.epochs):

        loss = trainer.train_epoch()

        print(
            f"Epoch {epoch + 1}/{config.training.epochs}"
            f" | Loss: {loss:.4f}"
        )

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": config.model.embedding_dimension,
        "context_length": config.model.context_length,
        "num_heads": config.model.number_of_heads,
        "num_layers": config.model.number_of_layers,
        "dropout": config.model.dropout,
    }

    torch.save(
        checkpoint,
        "checkpoints/tinygpt.pt",
    )

    print()

    print("Training Complete!")

    print("Checkpoint saved to checkpoints/tinygpt.pt")


if __name__ == "__main__":
    main()