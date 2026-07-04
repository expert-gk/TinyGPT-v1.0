from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Handles the training of a GPT model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:

        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Average training loss.
        """

        self.model.train()

        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc="Training",
        )

        for input_ids, target_ids in progress:

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward
            logits = self.model(input_ids)

            if torch.isnan(logits).any():
                raise RuntimeError("NaN detected in model output.")

            # (B,T,V) -> (B*T,V)
            vocab_size = logits.size(-1)

            loss = self.criterion(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
            )

            if torch.isnan(loss):
                raise RuntimeError("NaN detected in loss.")

            # Backward
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return total_loss / len(self.train_loader)