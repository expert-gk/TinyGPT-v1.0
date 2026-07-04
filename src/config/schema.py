from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProjectConfig:
    name: str
    version: str


@dataclass(slots=True)
class HardwareConfig:
    device: str


@dataclass(slots=True)
class DatasetConfig:
    train_file: str
    validation_file: str


@dataclass(slots=True)
class TokenizerConfig:
    type: str
    vocabulary_size: int | None


@dataclass(slots=True)
class ModelConfig:
    context_length: int
    embedding_dimension: int
    number_of_heads: int
    number_of_layers: int
    dropout: float


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float


@dataclass(slots=True)
class LoggingConfig:
    directory: str


@dataclass(slots=True)
class CheckpointConfig:
    directory: str
    save_every_epoch: bool


@dataclass(slots=True)
class GPTConfig:

    project: ProjectConfig

    hardware: HardwareConfig

    dataset: DatasetConfig

    tokenizer: TokenizerConfig

    model: ModelConfig

    training: TrainingConfig

    logging: LoggingConfig

    checkpoint: CheckpointConfig