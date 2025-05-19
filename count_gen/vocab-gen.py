from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import os

# Step 1: Read the corpus file
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.readlines()
    return corpus

# Step 2: Create an empty Hugging Face tokenizer
def create_tokenizer():
    # Use a Unigram model (a popular subword tokenization model)
    tokenizer = Tokenizer(models.Unigram())
    
    # Split text on whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    return tokenizer

# Step 3: Train the tokenizer and build the vocabulary
def train_tokenizer(corpus, tokenizer, vocab_size=5000):
    # Trainer configuration
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # Start training
    tokenizer.train_from_iterator(corpus, trainer)

    return tokenizer

# Step 4: Save the trained tokenizer to the project folder
def save_tokenizer(tokenizer, tokenizer_folder="tokenizer"):
    # Persist tokenizer files to the specified directory
    tokenizer.save(f"./{tokenizer_folder}")

# Usage example
file_path = "./c.txt"  # Path to the corpus file
corpus = read_corpus(file_path)
tokenizer = create_tokenizer()
tokenizer = train_tokenizer(corpus, tokenizer)

# Save the tokenizer
save_tokenizer(tokenizer, "tokenizer")

# Print the first 10 entries of the generated vocabulary
vocab = tokenizer.get_vocab()
print({k: vocab[k] for k in list(vocab)[:10]})
