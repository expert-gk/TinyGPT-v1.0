import torch

from src.models.embedding import InputEmbedding


def test_embedding() -> None:

    model = InputEmbedding(
        vocab_size=100,
        embedding_dim=128,
        context_length=64,
        dropout=0.1,
    )

    input_ids = torch.randint(
        0,
        100,
        (2, 64),
    )

    output = model(input_ids)

    assert output.shape == (2, 64, 128)