import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
from tokenizers import Tokenizer  # Ensure the tokenizers library is installed and configured
import random
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
class Config:
    def __init__(self):
        self.vocab_size = 574
        self.d_model = 256
        self.d_ff = 512
        self.num_heads = 16
        self.num_layers = 6
        self.dropout = 0.1
        self.max_len = 128
        # Use RoPE for the first k heads; others have no positional encoding
        self.k_rope_heads = 0   # Default: none
        self.num_epochs = 100
        self.weight_decay = 1e-5
        self.momentum_1 = 0.98
        self.momentum_2 = 0.9
        self.batch_size = 512

# -------------------------------
# RoPE Positional Encoding
# -------------------------------
class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int):
        """
        head_dim: dimension per head (must be even)
        max_len: maximum sequence length to pre-allocate
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim

        # 1. Compute inverse frequency
        half_dim = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float) / half_dim))
        # 2. Generate [max_len, half_dim] angle matrix
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        angle = positions * inv_freq.unsqueeze(0)                        # [max_len, half_dim]
        # 3. Expand to [max_len, head_dim] and cache
        #    Note: interleaved even/odd positions
        cos = torch.zeros(max_len, head_dim)
        sin = torch.zeros(max_len, head_dim)
        cos[:, 0::2] = torch.cos(angle)
        cos[:, 1::2] = torch.cos(angle)
        sin[:, 0::2] = torch.sin(angle)
        sin[:, 1::2] = torch.sin(angle)
        self.register_buffer("cos", cos)    # [max_len, head_dim]
        self.register_buffer("sin", sin)    # [max_len, head_dim]

    def forward(self, x: torch.Tensor, k: int = None) -> torch.Tensor:
        """
        x: Tensor of shape [B, H, L, D]
        k: number of heads to apply RoPE to; if None, apply to all
        Returns a tensor of the same shape
        """
        B, H, L, D = x.shape
        assert D == self.head_dim, f"Last dimension must match head_dim ({self.head_dim})"
        if k is None:
            k = H

        # Apply to first k heads
        x_rope = x[:, :k, :, :]            # [B, k, L, D]
        x_flat = x_rope.reshape(-1, L, D)  # [B*k, L, D]

        # Retrieve sin/cos for current sequence length
        cos = self.cos[:L].unsqueeze(0)    # [1, L, D]
        sin = self.sin[:L].unsqueeze(0)

        # Split even/odd and apply rotation
        x1 = x_flat[..., 0::2]             # [B*k, L, D/2]
        x2 = x_flat[..., 1::2]
        xr_even = x1 * cos[..., 0::2] - x2 * sin[..., 0::2]
        xr_odd  = x1 * sin[..., 0::2] + x2 * cos[..., 0::2]

        # Interleave back
        out = torch.empty_like(x_flat)
        out[..., 0::2] = xr_even
        out[..., 1::2] = xr_odd

        # Reshape back to [B, k, L, D]
        out = out.view(B, k, L, D)
        # Concatenate remaining heads
        if k < H:
            out = torch.cat([out, x[:, k:, :, :]], dim=1)
        return out

# -------------------------------
# Attention Decoder Layer
# -------------------------------
class AttentionDecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model       = config.d_model
        self.num_heads     = config.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim      = self.d_model // self.num_heads
        self.k_rope_heads  = config.k_rope_heads

        self.q_linear      = nn.Linear(self.d_model, self.d_model)
        self.k_linear      = nn.Linear(self.d_model, self.d_model)
        self.v_linear      = nn.Linear(self.d_model, self.d_model)
        self.out_linear    = nn.Linear(self.d_model, self.d_model)

        # RoPE module injected into multi-head attention
        self.rope          = RoPE(self.head_dim, config.max_len)

        self.ffn           = nn.Sequential(
            nn.Linear(self.d_model, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, self.d_model)
        )
        self.dropout       = nn.Dropout(config.dropout)
        self.norm1         = nn.LayerNorm(self.d_model)
        self.norm2         = nn.LayerNorm(self.d_model)

    def forward(self, x, memory=None):
        # 1. LayerNorm + linear projections
        context = x if memory is None else memory
        x_norm  = self.norm1(x)
        q       = self.q_linear(x_norm)
        k       = self.k_linear(context)
        v       = self.v_linear(context)

        # 2. Split heads → (B, H, L, D)
        B, L, _ = q.size()
        H, D    = self.num_heads, self.head_dim
        q = q.view(B, L, H, D).transpose(1, 2)
        k = k.view(B, L, H, D).transpose(1, 2)
        v = v.view(B, L, H, D).transpose(1, 2)

        # 3. Apply RoPE to first k_rope_heads heads
        q = self.rope(q, k=self.k_rope_heads)
        k = self.rope(k, k=self.k_rope_heads)

        # 4. Scale & flatten → (B*H, L, D)
        q = q * (1.0 / math.sqrt(D))
        q_flat = q.reshape(-1, L, D)
        k_flat = k.reshape(-1, L, D)
        v_flat = v.reshape(-1, L, D)

        # 5. Compute attention
        scores       = torch.matmul(q_flat, k_flat.transpose(-2, -1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out_flat     = torch.matmul(attn_weights, v_flat)

        # 6. Merge heads and project back to d_model
        attn_output = out_flat.view(B, H, L, D) \
                              .transpose(1, 2) \
                              .reshape(B, L, self.d_model)
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout(attn_output)
        x = x + attn_output

        # 7. Feed-forward + residual
        x_norm2    = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output

        return x

# -------------------------------
# Transformer Decoder
# -------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            AttentionDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tgt, memory=None):
        # Embedding
        tgt_embedded = self.embedding(tgt)
        memory_embedded = tgt_embedded if memory is None else self.embedding(memory)
        x = tgt_embedded
        for layer in self.layers:
            x = layer(x, memory_embedded)

        x = x.mean(dim=1)
        return self.fc_out(x)

# -------------------------------
# Dataset and Training/Evaluation
# -------------------------------
class GeneratedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer.encode(item['text'])
        ids = torch.tensor(encoding.ids, dtype=torch.long)
        label = torch.tensor(item['distance'], dtype=torch.long)
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        pad_len = self.max_length - ids.size(0)
        ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        return {'input_ids': ids, 'distance': label}

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        labels = batch['distance'].to(device)
        optimizer.zero_grad()
        output = model(ids, memory=ids)
        loss = criterion(output, labels)
        total_loss += loss.item()
        pred = output.argmax(dim=-1)
        mask = labels != 0
        correct += (pred[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            labels = batch['distance'].to(device)
            output = model(ids, memory=ids)
            loss = criterion(output, labels)
            total_loss += loss.item()
            pred = output.argmax(dim=-1)
            mask = labels != 0
            correct += (pred[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    return total_loss / len(loader), correct / total

def train_model(model, train_loader, test_loader, config, exp_dir):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=config.weight_decay,
        betas=(config.momentum_1, config.momentum_2)
    )
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_accs, test_accs = [], []
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch+1}/{config.num_epochs}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'accuracy_results.csv'), 'w') as f:
        f.write('Epoch,TrainAcc,TestAcc\n')
        for i, (ta, va) in enumerate(zip(train_accs, test_accs), 1):
            f.write(f"{i},{ta:.4f},{va:.4f}\n")

    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))

    plt.figure()
    plt.plot(range(1, config.num_epochs+1), train_accs, label='Train')
    plt.plot(range(1, config.num_epochs+1), test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'accuracy_plot.png'))
    plt.close()

    return train_accs, test_accs

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = Tokenizer.from_file('./tokenizer')
    data = load_dataset('generated_dataset.json')
    config = Config()

    dataset = GeneratedDataset(data, tokenizer, config.max_len)
    train_size = int(0.7 * len(dataset))
    g = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = random_split(dataset, [train_size, len(dataset) - train_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    k_val = [16]
    for k in k_val:
        config.k_rope_heads = k
        print(f"Training with k_rope_heads={k}")
        model = TransformerDecoder(config)
        exp_dir = os.path.join('result', f'rope-{k}-nope')
        train_model(model, train_loader, test_loader, config, exp_dir)
