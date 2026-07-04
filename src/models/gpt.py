from __future__ import annotations

import torch
import torch.nn as nn

from src.models.embedding import InputEmbedding
from src.models.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    Decoder-only GPT model.

    Architecture:

        Input IDs
            │
            ▼
        Input Embedding
            │
            ▼
        N × Transformer Blocks
            │
            ▼
        Final LayerNorm
            │
            ▼
        Language Model Head
            │
            ▼
        Vocabulary Logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_length=context_length,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            embedding_dim,
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

        # Weight tying (used by GPT-2 and later models)
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:
                Shape -> (B, T)

        Returns:
            logits:
                Shape -> (B, T, vocab_size)
        """

        _, sequence_length = input_ids.shape

        if sequence_length > self.embedding.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"context length ({self.embedding.position_embedding.num_embeddings})."
            )

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits