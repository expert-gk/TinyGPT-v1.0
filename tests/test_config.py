from config.config import Config


config = Config.load("configs/default.yaml")

print(config.get("project.name"))

print(config.get("model.embedding_dimension"))

print(config.get("training.batch_size"))

print(config.get("hardware.device"))