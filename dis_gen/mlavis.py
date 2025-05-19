import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from mla_dis import Config, MLATransformerDecoder, GeneratedDataset, evaluate  # assume evaluate() 返回 (loss, accuracy)

def mkdir(p):
    os.makedirs(p, exist_ok=True)

def visualize_mla_heads(model_path: str,
                        tokenizer_path: str,
                        data_path: str,
                        max_len: int,
                        batch_size: int = 1024,
                        device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    data = __import__('mla_dis').load_dataset(data_path)
    dataset = GeneratedDataset(data, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    config = Config()
    model  = MLATransformerDecoder(config)
    ckpt   = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    root = "vis/mla"
    mkdir(root)

    _, base_acc = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
    print(f"[baseline] acc = {base_acc:.4f}")
    with open(os.path.join(root, "baseline.txt"), "w") as f:
        f.write(f"{base_acc:.6f}\n")

    n_layers = len(model.layers)
    n_heads  = config.num_heads
    d_head   = config.d_model // n_heads

    for layer_idx in range(n_layers):
        layer_dir = os.path.join(root, f"layer_{layer_idx}")
        mkdir(layer_dir)

        attn = model.layers[layer_idx].attn
        Wuq_w = attn.W_UQ.weight.data.clone()
        Wuk_w = attn.W_UK.weight.data.clone()
        Wuv_w = attn.W_UV.weight.data.clone()

        head_accs = []
        for h in range(n_heads):
            attn.W_UQ.weight.data.copy_(Wuq_w)
            attn.W_UK.weight.data.copy_(Wuk_w)
            attn.W_UV.weight.data.copy_(Wuv_w)

            start, end = h * d_head, (h + 1) * d_head
            attn.W_UQ.weight.data[:, start:end] = 0.
            attn.W_UK.weight.data[:, start:end] = 0.
            attn.W_UV.weight.data[:, start:end] = 0.

            _, acc = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
            head_accs.append(acc)
            print(f"layer {layer_idx}, head {h:2d} → acc: {acc:.4f}")

        with open(os.path.join(layer_dir, "head_accs.txt"), "w") as f:
            for h, acc in enumerate(head_accs):
                f.write(f"{h}\t{acc:.6f}\n")

        plt.figure(figsize=(6,3))
        plt.plot(list(range(n_heads)), head_accs, marker='o')
        plt.title(f"MLA layer {layer_idx} head ablation")
        plt.xlabel("head index")
        plt.ylabel("accuracy")
        plt.xticks(range(n_heads))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(layer_dir, "head_acc_plot.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    visualize_mla_heads(
        model_path     = "./result/mla/rep_1/model.pth",
        tokenizer_path = "./tokenizer",
        data_path      = "generated_dataset.json",
        max_len        = 128,
        batch_size     = 1024
    )
