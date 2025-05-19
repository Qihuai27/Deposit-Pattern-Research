import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import os
import matplotlib.pyplot as plt
from strans_distrans_dis import Config, TransformerDecoder, load_dataset, GeneratedDataset

@torch.no_grad()
def evaluate_loader(model, loader, device):
    """
    Batch evaluation using the same mask & token-level accuracy as in training.
    """
    model.eval()
    total, correct = 0, 0
    for batch in loader:
        ids    = batch['input_ids'].to(device)    # [B, L]
        labels = batch['distance'].to(device)     # [B]
        logits = model(ids, memory=ids)           # [B, C]
        preds  = logits.argmax(dim=-1)            # [B]
        # Only count non-padding positions
        mask = (labels != 0)
        correct += (preds[mask] == labels[mask]).sum().item()
        total   += mask.sum().item()
    return correct / total

def mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def run_experiments(
    ckpt_pattern: str,
    tokenizer_path: str,
    data_path: str,
    max_len: int,
    batch_size: int = 1024,
    device: torch.device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preload tokenizer and dataset
    tokenizer    = Tokenizer.from_file(tokenizer_path)
    raw_data     = load_dataset(data_path)
    eval_dataset = GeneratedDataset(raw_data, tokenizer, max_len)
    eval_loader  = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    ks = [0, 1, 2, 3]
    for k in ks:
        print(f"\n=== Running for k = {k} ===")
        # Build the checkpoint path for this k
        ckpt_path = ckpt_pattern.format(k)
        if not os.path.isfile(ckpt_path):
            print(f"  [Warning] checkpoint not found: {ckpt_path}, skip k={k}")
            continue

        # Create directory for this k
        k_dir = os.path.join("vis", f"k_{k}")
        mkdir(k_dir)

        # 1. Initialize model
        config = Config()
        config.k_rope_heads = k
        model = TransformerDecoder(config)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        # 2. Baseline evaluation
        baseline_acc = evaluate_loader(model, eval_loader, device)
        print(f"[k={k}] Baseline accuracy: {baseline_acc:.4f}")

        # Save baseline result
        with open(os.path.join(k_dir, "baseline.txt"), "w") as f:
            f.write(f"{baseline_acc:.6f}\n")

        # 3. Zero-out each head in each layer
        num_layers = len(model.layers)
        num_heads  = config.num_heads
        head_dim   = config.d_model // num_heads

        for layer_idx in range(num_layers):
            layer_dir = os.path.join(k_dir, f"layer_{layer_idx}")
            mkdir(layer_dir)

            layer = model.layers[layer_idx]
            orig_w = layer.v_linear.weight.data.clone()
            orig_b = (
                layer.v_linear.bias.data.clone()
                if layer.v_linear.bias is not None else None
            )

            head_accs = []
            for h in range(num_heads):
                # Restore original weights
                layer.v_linear.weight.data.copy_(orig_w)
                if orig_b is not None:
                    layer.v_linear.bias.data.copy_(orig_b)

                # Zero-out head h
                start, end = h * head_dim, (h + 1) * head_dim
                layer.v_linear.weight.data[start:end, :].zero_()
                if orig_b is not None:
                    layer.v_linear.bias.data[start:end].zero_()

                # Evaluate
                acc = evaluate_loader(model, eval_loader, device)
                head_accs.append(acc)
                print(f"  layer {layer_idx}, head {h:2d} → acc: {acc:.4f}")

            # Save this layer’s head accuracies
            with open(os.path.join(layer_dir, "head_accs.txt"), "w") as f:
                for h, acc in enumerate(head_accs):
                    f.write(f"{h}\t{acc:.6f}\n")

            # Plot and save
            plt.figure(figsize=(9, 6))
            plt.plot(range(num_heads), head_accs, marker='o', linestyle='-')
            plt.title(f"k={k}  Layer={layer_idx} Head Accuracy")
            plt.xlabel("Head index")
            plt.ylabel("Accuracy")
            plt.xticks(range(num_heads))
            plt.grid(True)
            plt.tight_layout()
            img_path = os.path.join(layer_dir, "head_acc_plot.png")
            plt.savefig(img_path, dpi=200)
            plt.close()

if __name__ == "__main__":
    # The ckpt_pattern will substitute {k} with 0..3
    CKPT_PATTERN   = "./result/rope-{}-nope/model.pth"
    TOKENIZER_PATH = "./tokenizer"
    DATA_PATH      = "generated_dataset.json"
    MAX_LEN        = 128
    BATCH_SIZE     = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_experiments(
        ckpt_pattern=CKPT_PATTERN,
        tokenizer_path=TOKENIZER_PATH,
        data_path=DATA_PATH,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        device=device,
    )
