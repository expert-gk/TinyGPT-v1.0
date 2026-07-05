# TinyGPT v1.0

TinyGPT is an educational implementation of a decoder-only GPT model built entirely from scratch using PyTorch.

The goal of this project is to understand how GPT models work internally by implementing every major component without relying on high-level transformer libraries such as Hugging Face Transformers. Rather than using pre-built abstractions, TinyGPT implements the tokenizer, dataset pipeline, Transformer architecture, training loop, and text generation from scratch. The result is a small but fully functional GPT model that serves as a foundation for future improvements.

## Features
- CPU and CUDA support
- YAML-based configuration
- Word-level tokenizer
- Sliding window dataset pipeline
- PyTorch Dataset and DataLoader
- Multi-Head Self-Attention
- Transformer blocks
- GPT model
- Model training
- Checkpoint saving
- Text generation

## Architecture
```text
Raw Text
    │
    ▼
Tokenizer
    │
    ▼
Dataset
    │
    ▼
GPT Model
    │
    ▼
Training
    │
    ▼
Text Generation
```

## Current Limitations

TinyGPT v1.0 is intentionally simple.

Current limitations include:

- Word-level tokenizer
- Small training dataset
- Greedy decoding
- No validation dataset
- No mixed precision training
- No Flash Attention
- Learned positional embeddings (RoPE will be introduced in a future version)

## Design Highlights
- Modular architecture
- Configuration-driven
- Proper PyTorch Dataset
- DataLoader
- Fixed context window
- Production-style Multi-Head Attention
- Checkpoint saving
- Validation
- Logging
- Inference script
- Model generation

### How it is diffrent from [TinyLLM](https://github.com/expert-gk/TinyLLM)

| TinyLLM            | TinyGPT                                           |
| ------------------ | ------------------------------------------------- |
| Word tokenizer     | Start with a clean tokenizer, then upgrade to Byte Pair Encoding (BPE) |
| Simple dataset     | Proper `Dataset` + `DataLoader`                   |
| Manual testing     | Automated training pipeline                       |
| Single attention   | Production-style Multi-Head Attention             |
| No checkpoints     | Save and resume training                          |
| No validation      | Validation loss tracking                          |
| No logging         | Training metrics and logs                         |
| No text generation | Interactive inference                             |
| Educational        | Engineering-grade implementation                  |

---

## How to run this project

First check python version. It should be 3.11 or later
```
python --version
```

Check for nvidia GPU (Optional if you have)
```
nvidia-smi
```

Create a virtual environment
```
python -m venv .venv
```

Activate the virtual environment
```
.venv\Scripts\activate
```

Upgrade pip
```
python -m pip install --upgrade pip
```

Install only the absolute minimum to test GPU
```
pip install pyyaml
py -m tests.test_device  
```

Run a device detection script
- You can install CUDA packages if you see CUDA available 

Install required packages for CPU only
```
pip install torch
```

Install required packages for CUDA GPU only
```
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

Install other required packages
```
pip install tqdm
```

Install your project
```
pip install -e ".[dev]"
```

To save datasets to tokenizer.json
```
python main.py tokenizer train
```

Train the model
```
python train.py
```

Generation
```
python generate.py
```

pip install torchvision torchaudio numpy tiktoken tqdm pyyaml

---

## Project Structure

```text
TinyGPT/
│
├── configs/
│   └── default.yaml
│
├── datasets/
│   ├── train.txt
│   ├── val.txt
│   └── tokenizer.json
│
├── checkpoints/
│
├── src/
│   ├── cli/
│   ├── config/
│   ├── datasets/
│   ├── models/
│   ├── tokenizers/
│   ├── trainer/
│   ├── utils/
│   └── __init__.py
│
├── tests/
│
├── generate.py
├── train.py
├── main.py
├── pyproject.toml
├── README.md
└── LESSONS.md
```

### Folder and File Overview

#### `configs/`

Stores all project configuration files.

| File           | Purpose                                                                                                                                            |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `default.yaml` | Central configuration file containing model architecture, training parameters, dataset locations, device settings, and other configurable options. |

---

#### `datasets/`

Contains the training data and tokenizer vocabulary.

| File             | Purpose                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `train.txt`      | Training dataset used to build the tokenizer vocabulary and train the GPT model.                                 |
| `val.txt`        | Reserved for validation in future versions. It is not used in TinyGPT v1.0.                                      |
| `tokenizer.json` | Saved tokenizer vocabulary generated after tokenizer training. It is loaded during model training and inference. |

---

#### `checkpoints/`

Stores trained model checkpoints.

Typical contents:

* Model weights
* Training metadata
* Best model (future versions)

---

#### `src/`

Contains the complete TinyGPT source code.

---

### `src/cli/`

Command-line interface for running project commands.

| File          | Purpose                                             |
| ------------- | --------------------------------------------------- |
| `commands.py` | Implements CLI commands such as tokenizer training. |

---

### `src/config/`

Handles application configuration.

| File          | Purpose                                                  |
| ------------- | -------------------------------------------------------- |
| `config.py`   | Configuration-related helper functions.                  |
| `loader.py`   | Loads configuration files from disk.                     |
| `schema.py`   | Defines configuration schemas using Python data classes. |
| `settings.py` | Loads and validates project configuration.               |

---

### `src/datasets/`

Dataset implementation.

| File                  | Purpose                                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `language_dataset.py` | Converts raw text into fixed-length input and target sequences for GPT training using a sliding window approach. |

---

### `src/models/`

Core neural network implementation.

| File                   | Purpose                                                                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `embedding.py`         | Implements token embeddings and positional embeddings.                                                                                    |
| `attention.py`         | Implements causal Multi-Head Self-Attention.                                                                                              |
| `feed_forward.py`      | Implements the feed-forward network used inside Transformer blocks.                                                                       |
| `transformer_block.py` | Combines Layer Normalization, Multi-Head Attention, residual connections, and the feed-forward network into a complete Transformer block. |
| `gpt.py`               | Assembles the complete decoder-only GPT model.                                                                                            |

---

### `src/tokenizers/`

Tokenizer framework.

| File                   | Purpose                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `base.py`              | Abstract tokenizer interface.                                                                           |
| `word_tokenizer.py`    | Word-level tokenizer implementation used in TinyGPT v1.0.                                               |
| `trainer.py`           | Builds the tokenizer vocabulary from the training dataset and saves it to disk.                         |
| `tokenizer_factory.py` | Creates tokenizer instances, making it easier to support additional tokenizer types in future versions. |

---

### `src/trainer/`

Training pipeline.

| File           | Purpose                                                                                 |
| -------------- | --------------------------------------------------------------------------------------- |
| `trainer.py`   | Implements the training loop, loss calculation, backpropagation, and optimizer updates. |
| `evaluator.py` | Reserved for evaluation features in future versions.                                    |

---

### `src/utils/`

Utility functions shared across the project.

| File        | Purpose                                       |
| ----------- | --------------------------------------------- |
| `device.py` | Detects and configures CPU or CUDA execution. |
| `logger.py` | Provides centralized logging utilities.       |

---

### `tests/`

Contains unit tests for each major module.

Current test coverage includes:

* Configuration loading
* Device detection
* Tokenizer
* Dataset
* Embedding layer
* Multi-Head Attention
* Transformer block
* GPT model

---

### Root Files

| File             | Purpose                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| `train.py`       | Main entry point for model training.                                                                     |
| `generate.py`    | Loads a trained model and generates text from user prompts.                                              |
| `main.py`        | CLI entry point for project utilities such as tokenizer training.                                        |
| `pyproject.toml` | Defines project metadata, dependencies, and development tools.                                           |
| `README.md`      | Project documentation and setup guide.                                                                   |
| `LESSONS.md`     | Documents key concepts learned, implementation decisions, and challenges encountered during development. |

## Design Principles

TinyGPT v1.0 follows a modular architecture where each component has a single responsibility.

* **Configuration** is managed separately from the implementation.
* **Tokenization** is independent of the dataset pipeline.
* **Dataset** preparation is isolated from model architecture.
* **Model** components are modular and reusable.
* **Training** and **inference** are separate workflows.
* **Utilities** provide common functionality shared across the project.

This structure keeps the project easy to understand, test, maintain, and extend while remaining focused on learning how GPT models work internally.

## Test Description

The project includes a collection of unit tests to verify that each major component of TinyGPT works correctly. These tests are designed to validate individual modules before running the complete training pipeline.

| Test File                   | Purpose                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `test_config.py`            | Verifies that the YAML configuration is loaded correctly and all configuration values are available.                 |
| `test_device.py`            | Checks device detection and ensures the project can run on both CPU and CUDA-enabled GPUs.                           |
| `test_tokenizer.py`         | Validates tokenizer training, vocabulary creation, encoding, and decoding operations.                                |
| `test_dataset.py`           | Ensures the language dataset correctly creates input and target token sequences using the configured context length. |
| `test_embedding.py`         | Verifies the token embedding and positional embedding layers produce tensors with the expected dimensions.           |
| `test_attention.py`         | Tests the Multi-Head Self-Attention implementation, including tensor shapes and causal masking.                      |
| `test_transformer_block.py` | Confirms that a Transformer block processes the input correctly while preserving the expected output dimensions.     |
| `test_gpt.py`               | Performs an end-to-end forward pass through the GPT model and verifies the output logits have the correct shape.     |

### Running All Tests

Run the complete test suite:

```bash
pytest
```

Run an individual test:

```bash
pytest tests/test_gpt.py
```

or

```bash
pytest tests/test_attention.py
```

All tests should pass before training the model to ensure each component functions correctly.
