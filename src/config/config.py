from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Config:
    data: dict[str, Any]

    @classmethod
    def load(cls, config_path: str | Path) -> "Config":

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:

        value = self.data

        for item in key.split("."):

            if not isinstance(value, dict):
                return default

            value = value.get(item)

            if value is None:
                return default

        return value