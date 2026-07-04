import torch

from src.models.transformer_block import TransformerBlock


def test_transformer_block():

    model = TransformerBlock(
        embedding_dim=128,
        num_heads=4,
        context_length=64,
        dropout=0.1,
    )

    x = torch.randn(
        2,
        64,
        128,
    )

    y = model(x)

    assert y.shape == (2, 64, 128)