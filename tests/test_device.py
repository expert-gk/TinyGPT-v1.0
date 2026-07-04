from __future__ import annotations

import torch

from src.config.settings import load_config
from src.utils.device import get_device


def main() -> None:

    config = load_config("configs/default.yaml")

    device = get_device(config)

    print(f"Using device     : {device}")
    print(f"Torch version    : {torch.__version__}")
    print(f"CUDA version     : {torch.version.cuda}")
    print(f"CUDA available   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
