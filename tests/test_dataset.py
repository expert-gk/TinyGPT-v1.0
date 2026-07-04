from torch.utils.data import DataLoader

from src.config.settings import load_config
from src.datasets.language_dataset import LanguageDataset
from src.tokenizers.tokenizer_factory import TokenizerFactory


config = load_config(
    "configs/default.yaml"
)

tokenizer = TokenizerFactory.create(config)

dataset = LanguageDataset(
    config,
    tokenizer,
    config.dataset.train_file,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

for x, y in loader:

    print()

    print("Input Shape")

    print(x.shape)

    print()

    print("Target Shape")

    print(y.shape)

    print()

    print(x)

    print()

    print(y)

    break