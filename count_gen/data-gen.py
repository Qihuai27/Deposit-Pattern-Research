import random
import json

config = {
    "num_samples": 50000,         # 生成样本数
    "max_sentence_length": 120,    # 句子最大长度
    "num_triggers": 30             # 固定触发词数量
}

def load_vocab(file_path="vocab.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)
    return vocab

def select_trigger_words(vocab, num_triggers):
    return random.sample(list(vocab.keys()), num_triggers)

# 生成数据样本
def generate_data(vocab, config, trigger_words):
    data = []
    vocab_list = list(vocab.keys())

    for _ in range(config["num_samples"]):
        num_triggers_in_sentence = random.randint(0, min(100, config["max_sentence_length"]))

        triggers_in_sentence = [random.choice(trigger_words) for _ in range(num_triggers_in_sentence)]

        remaining_len = config["max_sentence_length"] - num_triggers_in_sentence
        non_triggers = [w for w in vocab_list if w not in trigger_words]
        filler_words = random.choices(non_triggers, k=max(0, remaining_len))

        sentence_words = triggers_in_sentence + filler_words
        random.shuffle(sentence_words)

        sample = {
            "text": " ".join(sentence_words),
            "distance": num_triggers_in_sentence
        }
        data.append(sample)

    return data

vocab = load_vocab("vocab.json")
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

trigger_words = select_trigger_words(vocab, config["num_triggers"])
print(f"Selected trigger words: {trigger_words}")

dataset = generate_data(vocab, config, trigger_words)

output_file = "generated_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"Dataset saved to {output_file}")