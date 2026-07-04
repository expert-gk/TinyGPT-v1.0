import torch

from src.config.settings import load_config
from src.models.attention import MultiHeadAttention


config = load_config("configs/default.yaml")

attention = MultiHeadAttention(
    embedding_dim=config.model.embedding_dimension,
    num_heads=config.model.number_of_heads,
    context_length=config.model.context_length,
    dropout=config.model.dropout,
)

x = torch.randn(
    2,
    config.model.context_length,
    config.model.embedding_dimension,
)

print("Input Shape :", x.shape)

y = attention(x)

print("Output Shape:", y.shape)