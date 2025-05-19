import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from tokenizers import Tokenizer  # Make sure the tokenizers library is installed and configured
import random
import numpy as np

# -------------------------------
# Configuration-related code
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
        # Options: 'absolute', 'relative', 'random', 'rope', 'nope', 'alibi'
        self.position_encoding_type = 'rope'  # Change default to 'alibi' if needed
        # Note: RoPE in attention is used only when position_encoding_type == 'rope'
        self.use_rope_in_attention = True
        self.num_epochs = 100
        self.weight_decay = 1e-5
        self.momentum_1 = 0.98
        self.momentum_2 = 0.9
        self.full_batch = False
        self.batch_size = 512

    def save(self):
        with open('config.json', 'w') as f:
            json.dump(self.__dict__, f)

# -------------------------------
# Positional encoding modules
# -------------------------------
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(AbsolutePositionalEncoding, self).__init__()
        self.position = torch.arange(0, max_len).unsqueeze(1)
        self.div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding[:, 0::2] = torch.sin(self.position * self.div_term)
        self.encoding[:, 1::2] = torch.cos(self.position * self.div_term)

    def forward(self, x):
        # Return positional encodings for the input sequence length
        return self.encoding[:x.size(1), :].unsqueeze(0).to(x.device)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(RelativePositionalEncoding, self).__init__()
        self.relative_position_embeddings = nn.Embedding(2 * max_len - 1, d_model)
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # Placeholder: returns zeros of appropriate shape
        return torch.zeros(batch_size, seq_len, self.relative_position_embeddings.embedding_dim, device=x.device)

class RandomPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(RandomPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.position_embeddings(positions)

class RoPE(nn.Module):
    def __init__(self, d_model, max_len, use_for_attention=False):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_for_attention = use_for_attention
        # Precompute inverse frequency for RoPE
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, d_model, 2, dtype=torch.float) / d_model))
        )

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.float, device=x.device)
        sinusoidal_pos = torch.einsum('i,j->ij', positions, self.inv_freq)
        sin, cos = torch.sin(sinusoidal_pos), torch.cos(sinusoidal_pos)
        rope_encoding = torch.cat([sin, cos], dim=-1)
        return rope_encoding.unsqueeze(0).expand(x.size(0), -1, -1)

    def apply_rope(self, tensor):
        batch_size, seq_len, d_model = tensor.size()
        positions = torch.arange(seq_len, dtype=torch.float, device=tensor.device)
        angles = positions[:, None] * self.inv_freq[None, :]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        tensor = tensor.view(batch_size, seq_len, d_model // 2, 2)
        tensor_rot = torch.stack([
            tensor[..., 0] * cos - tensor[..., 1] * sin,
            tensor[..., 0] * sin + tensor[..., 1] * cos
        ], dim=-1)
        return tensor_rot.view(batch_size, seq_len, d_model)

class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(NoPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # Return zeros (no positional information)
        return torch.zeros(batch_size, seq_len, self.d_model, device=x.device)

# -------------------------------
# TransformerDecoder and AttentionDecoderLayer
# -------------------------------
class AttentionDecoderLayer(nn.Module):
    def __init__(self, config):
        super(AttentionDecoderLayer, self).__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads

        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.out_linear = nn.Linear(config.d_model, config.d_model)

        if config.position_encoding_type == 'rope' and config.use_rope_in_attention:
            self.rope = RoPE(self.head_dim, config.max_len, use_for_attention=True)
        else:
            self.rope = None

        if config.position_encoding_type == 'relative':
            # Relative bias parameters
            self.relative_bias_table = nn.Parameter(
                torch.zeros(2 * config.max_len - 1, config.num_heads)
            )
            nn.init.normal_(self.relative_bias_table, std=0.02)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def compute_alibi_bias(self, seq_len, num_heads):
        # Compute ALiBi positional bias
        m = torch.exp(-torch.arange(0, num_heads).float() / num_heads)
        positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        bias = -torch.abs(positions).unsqueeze(0) * m.unsqueeze(1).unsqueeze(1)
        return bias  # shape [num_heads, seq_len, seq_len]

    def forward(self, x, memory=None):
        context = x if memory is None else memory

        # Pre-LN self-attention
        x_norm = self.norm1(x)
        q = self.q_linear(x_norm)
        k = self.k_linear(context)
        v = self.v_linear(context)

        batch_size, seq_len, _ = q.size()
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Flatten head dimensions
        q_flat = q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k_flat = k.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        v_flat = v.reshape(batch_size * self.num_heads, seq_len, self.head_dim)

        # Apply RoPE if configured
        if self.rope is not None:
            q_flat = self.rope.apply_rope(q_flat)
            k_flat = self.rope.apply_rope(k_flat)

        # Restore shapes
        q = q_flat.view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k_flat.view(batch_size, self.num_heads, seq_len, self.head_dim)
        q = q * (1.0 / (self.head_dim ** 0.5))

        q_flat = q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k_flat = k.reshape(batch_size * self.num_heads, seq_len, self.head_dim)

        # Compute attention scores
        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1))

        # Add positional biases if needed
        if self.config.position_encoding_type == 'alibi':
            alibi_bias = self.compute_alibi_bias(seq_len, self.num_heads).to(scores.device)
            scores = scores.view(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores + alibi_bias.unsqueeze(0)
            scores = scores.view(batch_size * self.num_heads, seq_len, seq_len)
        elif self.config.position_encoding_type == 'relative':
            # Compute relative position indices
            rel_pos = torch.arange(seq_len, device=q.device).unsqueeze(0) - \
                      torch.arange(seq_len, device=q.device).unsqueeze(1)
            rel_pos += self.config.max_len - 1
            relative_bias = self.relative_bias_table[rel_pos].permute(2, 0, 1)
            scores = scores.view(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores + relative_bias.unsqueeze(0)
            scores = scores.view(batch_size * self.num_heads, seq_len, seq_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output_flat = torch.matmul(attn_weights, v_flat)
        attn_output = attn_output_flat.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.out_linear(attn_output)
        attn_output = self.dropout(attn_output)
        x = x + attn_output

        # Pre-LN feed-forward
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        ffn_output = self.dropout(ffn_output)
        x = x + ffn_output
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_encoding = self.get_position_encoding(config)
        self.layers = nn.ModuleList([AttentionDecoderLayer(config) for _ in range(config.num_layers)])
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def get_position_encoding(self, config):
        if config.position_encoding_type == 'absolute':
            return AbsolutePositionalEncoding(config.d_model, config.max_len)
        elif config.position_encoding_type == 'relative':
            return NoPositionalEncoding(config.d_model, config.max_len)
        elif config.position_encoding_type == 'random':
            return RandomPositionalEncoding(config.d_model, config.max_len)
        elif config.position_encoding_type == 'rope':
            return RoPE(config.d_model, config.max_len, use_for_attention=False)
        elif config.position_encoding_type == 'nope':
            return NoPositionalEncoding(config.d_model, config.max_len)
        elif config.position_encoding_type == 'alibi':
            return NoPositionalEncoding(config.d_model, config.max_len)
        else:
            raise ValueError(f"Unsupported position encoding type: {config.position_encoding_type}")

    def forward(self, tgt, memory=None):
        # Embedding + (optional) positional encoding
        if self.config.position_encoding_type == 'nope':
            tgt_embedded = self.embedding(tgt)
        else:
            pos_encoding = self.position_encoding(tgt)
            tgt_embedded = self.embedding(tgt) + pos_encoding

        if memory is None:
            memory_embedded = tgt_embedded
        else:
            if memory.dtype == torch.long:
                mem_pos = self.position_encoding(memory)
                memory_embedded = self.embedding(memory) + mem_pos
            else:
                memory_embedded = memory

        x = tgt_embedded
        for layer in self.layers:
            x = layer(x, memory_embedded)

        x = torch.mean(x, dim=1)  # Pool across sequence
        output = self.fc_out(x)
        return output

# -------------------------------
# Dataset and loading
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
        data = json.load(f)
    return data

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        # Use math.cos and math.pi for cosine schedule
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------------------
# Training and evaluation
# -------------------------------
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

def train_model(model, train_loader, test_loader, num_epochs=10,
                learning_rate=1e-4, exp_dir='./result/default'):
    device = torch.device(
        'cuda:1' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'cuda:0'
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.momentum_1, config.momentum_2)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, scheduler, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}"
        )

    os.makedirs(exp_dir, exist_ok=True)
    result_file = os.path.join(exp_dir, 'accuracy_results.csv')
    with open(result_file, 'w') as f:
        f.write("Epoch,Train Accuracy,Test Accuracy\n")
        for i in range(num_epochs):
            f.write(f"{i+1},{train_accuracies[i]:.4f},{test_accuracies[i]:.4f}\n")
    print(f"Accuracy results saved to {result_file}")

    # Save model checkpoint
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
# Batch experiment entry point
# -------------------------------
if __name__ == '__main__':
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load tokenizer and dataset once globally
    tokenizer = Tokenizer.from_file('./tokenizer')
    data = load_dataset('generated_dataset.json')

    # Ensure result directory exists
    os.makedirs('./result', exist_ok=True)

    # Different positional encoding types to experiment with
    experiment_list = ['nope', 'absolute', 'alibi', 'relative', 'random', 'rope']
    global_results = []
    repeat_times = 1

    for encoding in experiment_list:
        print(f"\n========== Running Experiment for position_encoding_type = {encoding} ==========")
        rep_results = []
        for rep in range(repeat_times):
            print(f"\n--- Repetition {rep+1}/{repeat_times} for {encoding} ---")
            # Instantiate new config and set encoding
            config = Config()
            config.position_encoding_type = encoding
            config.save()  # Save current experiment config
          
            # Prepare dataset and DataLoader
            dataset = GeneratedDataset(data, tokenizer, config.max_len)
            train_size = int(0.7 * len(dataset))
            test_size = len(dataset) - train_size
            g = torch.Generator().manual_seed(SEED)
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=g)
            if config.full_batch:
                train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
            else:
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

            # Initialize model and run training
            model = TransformerDecoder(config)
            exp_dir = os.path.join('./result', encoding, f'rep_{rep+1}')
            train_acc, test_acc = train_model(
                model, train_loader, test_loader,
                num_epochs=config.num_epochs,
                learning_rate=1e-4, exp_dir=exp_dir
            )
            rep_results.append({
                'final_train_accuracy': train_acc[-1],
                'final_test_accuracy': test_acc[-1]
            })
        
        # Compute average metrics for this encoding
        avg_train = sum(r['final_train_accuracy'] for r in rep_results) / repeat_times
        avg_test  = sum(r['final_test_accuracy'] for r in rep_results) / repeat_times
        global_results.append({
            'experiment': encoding,
            'avg_train_accuracy': avg_train,
            'avg_test_accuracy': avg_test,
            'rep_results': rep_results
        })
    
    # Save summary of all experiments
    summary_file = os.path.join('./result', 'experiment_summary.csv')
    with open(summary_file, 'w') as f:
        header = "Experiment," + ",".join(
            [f"Rep{i+1}_Train,Rep{i+1}_Test" for i in range(repeat_times)]
        ) + ",Avg_Train,Avg_Test\n"
        f.write(header)
        for res in global_results:
            line = res['experiment'] + ","
            for rep in res['rep_results']:
                line += f"{rep['final_train_accuracy']:.4f},{rep['final_test_accuracy']:.4f},"
            line += f"{res['avg_train_accuracy']:.4f},{res['avg_test_accuracy']:.4f}\n"
            f.write(line)
    print(f"\nGlobal experiment summary saved to {summary_file}")
