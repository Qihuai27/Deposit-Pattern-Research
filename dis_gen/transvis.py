# trans_eval.py
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from trans_dis import (                      
    Config, TransformerDecoder,
    GeneratedDataset, load_dataset, evaluate # evaluate(model, loader, crit, device)
)

def mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def evaluate_and_ablate(model, loader, device, root_dir):
    crit = torch.nn.CrossEntropyLoss(ignore_index=0)

    # baseline
    _, base_acc = evaluate(model, loader, crit, device)
    with open(os.path.join(root_dir, "baseline.txt"), "w") as f:
        f.write(f"{base_acc:.6f}\n")
    print(f"  baseline acc = {base_acc:.4f}")

    # head‑ablation
    n_layers = len(model.layers)
    n_heads  = model.config.num_heads
    d_head   = model.config.d_model // n_heads

    for layer_idx, layer in enumerate(model.layers):
        ldir = os.path.join(root_dir, f"layer_{layer_idx}")
        mkdir(ldir)
        w_backup = layer.v_linear.weight.data.clone()
        b_backup = (
            layer.v_linear.bias.data.clone()
            if layer.v_linear.bias is not None else None
        )

        head_accs = []
        for h in range(n_heads):
            layer.v_linear.weight.data.copy_(w_backup)
            if b_backup is not None:
                layer.v_linear.bias.data.copy_(b_backup)
            s, e = h * d_head, (h + 1) * d_head
            layer.v_linear.weight.data[s:e, :].zero_()
            if b_backup is not None:
                layer.v_linear.bias.data[s:e].zero_()

            _, acc = evaluate(model, loader, crit, device)
            head_accs.append(acc)
            print(f"    L{layer_idx}‑H{h:02d}: {acc:.4f}")

        with open(os.path.join(ldir, "head_accs.txt"), "w") as f:
            for h, acc in enumerate(head_accs):
                f.write(f"{h}\t{acc:.6f}\n")

        plt.figure(figsize=(5.2, 3))
        plt.plot(range(n_heads), head_accs, marker='o')
        plt.title(f"L{layer_idx} head ablation")
        plt.xlabel("head")
        plt.ylabel("accuracy")
        plt.xticks(range(n_heads))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(ldir, "head_acc_plot.png"), dpi=200)
        plt.close()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ENCODINGS   = ['nope', 'absolute', 'alibi', 'relative', 'random', 'rope']
    REPEAT_TIMES = 1  
    CKPT_PATTERN = "./result/{enc}/rep_{rep}/model.pth"

    tokenizer = Tokenizer.from_file("./tokenizer")
    full_data = load_dataset("generated_dataset.json")
    dataset   = GeneratedDataset(full_data, tokenizer, Config().max_len)
    loader    = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    for enc in ENCODINGS:
        for rep in range(1, REPEAT_TIMES + 1):
            ckpt = CKPT_PATTERN.format(enc=enc, rep=rep)
            if not os.path.isfile(ckpt):
                print(f"[skip] {ckpt} not found")
                continue

            print(f"\n=== {enc} — rep {rep} ===")
            # init matching config & model
            cfg = Config()
            cfg.position_encoding_type = enc
            model = TransformerDecoder(cfg)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.to(DEVICE).eval()

            out_root = f"vis_trans/{enc}/rep_{rep}"
            mkdir(out_root)
            evaluate_and_ablate(model, loader, DEVICE, out_root)

if __name__ == "__main__":
    main()
