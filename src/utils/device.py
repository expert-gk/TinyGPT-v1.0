from __future__ import annotations

import torch

from src.config.schema import GPTConfig


def get_device(config: GPTConfig) -> torch.device:

    device = config.hardware.device.lower()

    if device == "cpu":
        return torch.device("cpu")

    if device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is unavailable.")

        return torch.device("cuda")

    if device == "auto":

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {device}"
    )