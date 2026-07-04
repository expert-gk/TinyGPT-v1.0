from __future__ import annotations

import torch

from src.config.settings import load_config
from src.models.gpt import GPTModel
from src.tokenizers.word_tokenizer import WordTokenizer
from src.utils.device import get_device


def load_model():

    config = load_config("configs/default.yaml")

    device = get_device(config)

    tokenizer = WordTokenizer.load(
        "datasets/tokenizer.json"
    )

    checkpoint = torch.load(
        "checkpoints/tinygpt.pt",
        map_location=device,
    )

    model = GPTModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        context_length=checkpoint["context_length"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, tokenizer, config, device


def generate(
    model,
    tokenizer,
    config,
    device,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
):

    tokens = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the latest context window
            input_context = input_ids[
                :,
                -config.model.context_length :,
            ]

            logits = model(input_context)

            # Last token prediction
            logits = logits[:, -1, :]

            # Greedy decoding
            # next_token = torch.argmax(
            #     logits,
            #     dim=-1,
            #     keepdim=True,
            # )

            probabilities = torch.softmax(
                logits / temperature,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                1,
            )

            token_id = next_token.item()
            word = tokenizer.decode([token_id])

            # print(f"Predicted: {token_id} -> {word}")

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

    output_tokens = input_ids.squeeze().tolist()

    return tokenizer.decode(output_tokens)


def main():

    model, tokenizer, config, device = load_model()

    print()

    print("TinyGPT v1")

    print("------------------------")

    while True:

        prompt = input("\nPrompt: ").strip()

        if prompt.lower() in ["exit", "quit"]:

            break

        response = generate(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            prompt=prompt,
        )

        print()

        print(response)


if __name__ == "__main__":

    main()