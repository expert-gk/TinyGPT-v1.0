from src.config.loader import load_yaml

from src.config.schema import *


def load_config(path: str):

    data = load_yaml(path)

    return GPTConfig(

        project=ProjectConfig(
            **data["project"]
        ),

        hardware=HardwareConfig(
            **data["hardware"]
        ),

        dataset=DatasetConfig(
            **data["dataset"]
        ),

        tokenizer=TokenizerConfig(
            **data["tokenizer"]
        ),

        model=ModelConfig(
            **data["model"]
        ),

        training=TrainingConfig(
            **data["training"]
        ),

        logging=LoggingConfig(
            **data["logging"]
        ),

        checkpoint=CheckpointConfig(
            **data["checkpoint"]
        )
    )