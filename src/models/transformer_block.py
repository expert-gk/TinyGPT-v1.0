from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Expands the embedding dimension by 4x and
    projects it back.

    GPT-2 Architecture:

        Linear
            ↓
        GELU
            ↓
        Linear
            ↓
        Dropout
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim * 4,
            ),
            nn.GELU(),
            nn.Linear(
                embedding_dim * 4,
                embedding_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


class TransformerBlock(nn.Module):
    """
    GPT Transformer Block

    Architecture

        LayerNorm
            ↓
        Multi-Head Attention
            ↓
        Residual Add
            ↓
        LayerNorm
            ↓
        Feed Forward
            ↓
        Residual Add
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.norm1 = nn.LayerNorm(
            embedding_dim,
        )

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim,
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Pre-Norm Attention
        x = x + self.attention(
            self.norm1(x)
        )

        # Pre-Norm Feed Forward
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x