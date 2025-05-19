import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
from tokenizers import Tokenizer

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
        # MLA-specific settings
        self.d_compress = 128
        self.position_encoding_type = 'rope'
        self.use_rope_in_attention = True
        self.num_epochs = 150
        self.weight_decay = 1e-5
        self.momentum_1 = 0.98
        self.momentum_2 = 0.9
        self.full_batch = False
        self.batch_size = 1024

    def save(self):
        with open('config.json', 'w') as f:
            json.dump(self.__dict__, f)


# -------------------------------
# Rotary Positional Embedding for MLA
# -------------------------------
class RotaryEmbedding(nn.Module):
    """RoPE for one head dimension (d_head)."""
    def __init__(self, d_head: int, max_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        self.register_buffer("rope", emb, persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        rope = self.rope.index_select(0, pos.to(self.rope.device))
        while rope.dim() < x.dim():
            rope = rope.unsqueeze(0)
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        sin, cos = rope[..., ::2], rope[..., 1::2]
        return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)


# -------------------------------
# Multi-Head Latent Attention
# -------------------------------
class MLAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_compress: int, max_len: int = 4096, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_DKV = nn.Linear(d_model, d_compress, bias=bias)
        self.W_DQ  = nn.Linear(d_model, d_compress, bias=bias)
        self.W_UK  = nn.Linear(d_compress, n_heads * self.d_head, bias=False)
        self.W_UV  = nn.Linear(d_compress, n_heads * self.d_head, bias=False)
        self.W_UQ  = nn.Linear(d_compress, n_heads * self.d_head, bias=False)
        self.W_KR  = nn.Linear(d_model, self.d_head, bias=bias)
        self.W_QR  = nn.Linear(d_compress, self.d_head, bias=bias)

        self.rope     = RotaryEmbedding(self.d_head, max_len=max_len)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = x.shape
        heads = self.n_heads
        device = x.device

        c_kv = self.W_DKV(x)
        k_c = self.W_UK(c_kv).view(B, L, heads, self.d_head)
        v_c = self.W_UV(c_kv).view(B, L, heads, self.d_head)

        k_r = self.W_KR(x)
        pos = torch.arange(L, device=device)
        k_r = self.rope(k_r, pos).unsqueeze(2).expand(-1, -1, heads, -1)
        k = torch.cat([k_c, k_r], dim=-1)

        c_q = self.W_DQ(x)
        q_c = self.W_UQ(c_q).view(B, L, heads, self.d_head)
        q_r = self.W_QR(c_q).view(B, L, 1, self.d_head)
        q_r = self.rope(q_r.squeeze(2), pos).unsqueeze(2).expand(-1, -1, heads, -1)
        q = torch.cat([q_c, q_r], dim=-1)

        q = q.permute(0, 2, 1, 3)        # [B, heads, L, d_head]
        k = k.permute(0, 2, 3, 1)        # [B, heads, d_head*2, L]
        v_c = v_c.permute(0, 2, 1, 3)    # [B, heads, L, d_head]

        scores = torch.matmul(q, k) / math.sqrt(2 * self.d_head)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v_c)

        ctx = ctx.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(ctx)


class MLADecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn  = MLAAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            d_compress=config.d_compress,
            max_len=config.max_len
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attention_mask=None):
        x_norm     = self.norm1(x)
        attn_output = self.attn(x_norm, attention_mask=attention_mask)
        x          = x + attn_output

        x_norm2    = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x          = x + ffn_output
        return x


# -------------------------------
# MLA Transformer Decoder
# -------------------------------
class MLATransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config   = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_len, config.d_model))
        self.layers    = nn.ModuleList([MLADecoderLayer(config) for _ in range(config.num_layers)])
        self.ln_f      = nn.LayerNorm(config.d_model)
        self.fc_out    = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tgt, attention_mask=None):
        B, L = tgt.size()
        tgt_embedded = self.embedding(tgt) + self.pos_embed[:, :L, :]
        x = tgt_embedded
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        # Masked mean pooling
        x = torch.sum(x * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return self.fc_out(x)


# -------------------------------
# Dataset and DataLoader
# -------------------------------
class GeneratedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        distance = item['distance']
        encoding = self.tokenizer.encode(text)
        input_ids = torch.tensor(encoding.ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        pad_len = self.max_length - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'distance': torch.tensor(distance, dtype=torch.long)
        }


def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------------
# Training and Evaluation
# -------------------------------
def train(model, train_loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['distance'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask=attention_mask)
        loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
        total_loss += loss.item()
        _, predicted = torch.max(output, dim=-1)
        correct += (predicted == labels).sum().item()
        total += labels.numel()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_loader), correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['distance'].to(device)
            output = model(input_ids, attention_mask=attention_mask)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            total_loss += loss.item()
            _, predicted = torch.max(output, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.numel()
    return total_loss / len(test_loader), correct / total


def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-4, exp_dir='./result/default'):
    device = torch.device('cuda:3' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda:0')
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.momentum_1, config.momentum_2)
    )
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train_accuracies, test_accuracies = [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'accuracy_results.csv'), 'w') as f:
        f.write("Epoch,Train Accuracy,Test Accuracy\n")
        for i in range(num_epochs):
            f.write(f"{i+1},{train_accuracies[i]:.4f},{test_accuracies[i]:.4f}\n")

    torch.save(model.state_dict(), os.path.join(exp_dir, 'model.pth'))

    figure_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(figure_dir, 'accuracy_plot.png'))
    plt.close()

    return train_accuracies, test_accuracies


# -------------------------------
# Batch Experiment Entry Point
# -------------------------------
if __name__ == '__main__':
    tokenizer = Tokenizer.from_file('./tokenizer')
    data = load_dataset('generated_dataset.json')

    os.makedirs('./result', exist_ok=True)

    experiment_list = ['mla']
    global_results = []
    repeat_times = 1

    for encoding in experiment_list:
        print(f"\n========== Running Experiment for model_type = {encoding} ==========")
        rep_results = []
        for rep in range(repeat_times):
            print(f"\n--- Repetition {rep+1}/{repeat_times} for {encoding} ---")
            config = Config()
            config.position_encoding_type = 'rope'
            config.save()

            dataset = GeneratedDataset(data, tokenizer, config.max_len)
            train_size = int(0.7 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            if config.full_batch:
                train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

            model = MLATransformerDecoder(config)
            exp_dir = os.path.join('./result', encoding, f'rep_{rep+1}')
            train_acc, test_acc = train_model(
                model, train_loader, test_loader,
                num_epochs=config.num_epochs,
                learning_rate=1e-4,
                exp_dir=exp_dir
            )
            rep_results.append({
                'final_train_accuracy': train_acc[-1],
                'final_test_accuracy': test_acc[-1]
            })

        avg_train = sum(r['final_train_accuracy'] for r in rep_results) / repeat_times
        avg_test  = sum(r['final_test_accuracy'] for r in rep_results) / repeat_times
        global_results.append({
            'experiment': encoding,
            'avg_train_accuracy': avg_train,
            'avg_test_accuracy': avg_test,
            'rep_results': rep_results
        })

    summary_file = os.path.join('./result', 'experiment_summary.csv')
    with open(summary_file, 'w') as f:
        header = "Experiment," + ",".join([f"Rep{i+1}_Train,Rep{i+1}_Test" for i in range(repeat_times)]) + ",Avg_Train,Avg_Test\n"
        f.write(header)
        for res in global_results:
            line = res['experiment'] + ","
            for rep in res['rep_results']:
                line += f"{rep['final_train_accuracy']:.4f},{rep['final_test_accuracy']:.4f},"
            line += f"{res['avg_train_accuracy']:.4f},{res['avg_test_accuracy']:.4f}\n"
            f.write(line)
    print(f"\nGlobal experiment summary saved to {summary_file}")
