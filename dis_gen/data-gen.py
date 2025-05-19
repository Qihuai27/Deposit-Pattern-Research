import random
import json
from pathlib import Path

# Default configuration for dataset generation
DEFAULT_CONFIG = {
    "num_samples": 100_000,        # Total number of samples to generate
    "max_sentence_length": 120,    # Maximum length of each generated sentence
    "max_words_between": 70,       # Maximum number of tokens between trigger words
    "max_words_side": 25,          # Maximum number of tokens on each side of trigger words
    # Will be set dynamically based on vocabulary size
    "vocab_size": None,
}

def load_vocab(file_path: str = "vocab.json") -> dict:
    """
    Load the vocabulary mapping from a JSON file.

    Args:
        file_path: Path to the vocabulary file.

    Returns:
        A dictionary mapping tokens to their metadata or indices.
    """
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def generate_trigger_pair(vocab: dict) -> tuple:
    """
    Randomly select two distinct trigger words from the vocabulary.

    Args:
        vocab: Dictionary of vocabulary tokens.

    Returns:
        A tuple containing two trigger words.
    """
    return tuple(random.sample(list(vocab.keys()), 2))


def generate_data(vocab: dict, config: dict) -> list:
    """
    Generate a dataset of sentences containing two trigger words
    separated and surrounded by random tokens.

    Args:
        vocab: Dictionary of vocabulary tokens.
        config: Configuration parameters for generation.

    Returns:
        A list of examples, each a dict with 'text' and 'distance'.
    """
    examples = []
    word1, word2 = generate_trigger_pair(vocab)

    for _ in range(config["num_samples"]):
        # Determine how many tokens appear between and around triggers
        between_count = random.randint(1, config["max_words_between"])
        if config["max_words_side"] > 0:
            left_count = random.randint(1, config["max_words_side"])
            right_count = random.randint(1, config["max_words_side"])
        else:
            left_count = right_count = 0

        # Sample random tokens for each segment
        left_tokens = random.sample(list(vocab.keys()), left_count)
        middle_tokens = random.sample(list(vocab.keys()), between_count)
        right_tokens = random.sample(list(vocab.keys()), right_count)

        # Assemble the sentence
        sentence = (
            left_tokens + [word1] + middle_tokens + [word2] + right_tokens
            if left_count > 0 else
            [word1] + middle_tokens + [word2]
        )

        # Truncate if exceeding max length
        if len(sentence) > config["max_sentence_length"]:
            sentence = sentence[: config["max_sentence_length"]]

        # Distance is number of tokens between triggers plus one
        distance = len(middle_tokens) + 1

        examples.append({
            "text": " ".join(sentence),
            "distance": distance,
        })

    # Remove duplicates while preserving content
    unique = {json.dumps(item): item for item in examples}
    return list(unique.values())


def main():
    # Load vocabulary and update configuration
    vocab = load_vocab("vocab.json")
    vocab_size = len(vocab)
    DEFAULT_CONFIG["vocab_size"] = vocab_size

    print(f"Loaded vocabulary with {vocab_size} entries.")

    # Generate dataset
    dataset = generate_data(vocab, DEFAULT_CONFIG)
    output_path = Path("generated_dataset.json")
    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(dataset, out_file, ensure_ascii=False, indent=4)

    print(f"Dataset generation complete: {len(dataset)} examples saved to {output_path}")


if __name__ == "__main__":
    main()
