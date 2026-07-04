import torch

from src.models.gpt import GPTModel


def test_gpt_forward() -> None:

    vocab_size = 100

    model = GPTModel(
        vocab_size=vocab_size,
        embedding_dim=128,
        context_length=64,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, 64),
    )

    logits = model(input_ids)

    assert logits.shape == (2, 64, vocab_size)