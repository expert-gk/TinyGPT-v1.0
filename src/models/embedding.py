from __future__ import annotations

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Creates the initial input embeddings for the GPT model.

    The output is:

        Token Embedding
            +
        Position Embedding
            ↓
         Dropout

    Input:
        (B, T)

    Output:
        (B, T, C)

    B = Batch Size
    T = Context Length
    C = Embedding Dimension
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:
                Shape -> (B, T)

        Returns:
            Tensor:
                Shape -> (B, T, C)
        """

        batch_size, sequence_length = input_ids.shape

        # Create position indices:
        # [0, 1, 2, ..., T-1]
        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        # Expand to every batch
        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length,
        )

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings