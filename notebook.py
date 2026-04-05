"""
Non-causal language modeling: Transformer model comparison.
This script is structured as a Kaggle-compatible notebook.
Predicts the central word given 3 previous + 3 next context words.
"""

# %% Imports and setup
from types import SimpleNamespace
import os
import pathlib
import pickle
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# %% Configuration
DATASET_VERSION = 'ca-100'
COMPETITION_ROOT = '../input/competitions/wordvectors-2026'
DATASET_ROOT = f'../input/notebooks/jarfo1/text-preprocessing/data/{DATASET_VERSION}'
WORKING_ROOT = f'data/{DATASET_VERSION}'
DATASET_PREFIX = 'ca.wiki'

params = SimpleNamespace(
    embedding_dim=256,
    window_size=7,
    batch_size=2048,
    epochs=4,
    preprocessed=f'{DATASET_ROOT}/{DATASET_PREFIX}',
    working=f'{WORKING_ROOT}/{DATASET_PREFIX}',
    modelname=f'{WORKING_ROOT}/{DATASET_VERSION}.pt',
    train=True,
)

# %% Vocabulary
class Vocabulary(object):
    def __init__(self, pad_token='<pad>', unk_token='<unk>', eos_token='<eos>'):
        self.token2idx = {}
        self.idx2token = []
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        if pad_token is not None:
            self.pad_index = self.add_token(pad_token)
        if unk_token is not None:
            self.unk_index = self.add_token(unk_token)
        if eos_token is not None:
            self.eos_index = self.add_token(eos_token)

    def add_token(self, token):
        if token not in self.token2idx:
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1
        return self.token2idx[token]

    def get_index(self, token):
        if isinstance(token, str):
            return self.token2idx.get(token, self.unk_index)
        else:
            return [self.token2idx.get(t, self.unk_index) for t in token]

    def get_token(self, index):
        return self.idx2token[index]

    def __len__(self):
        return len(self.idx2token)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

# %% Data utilities
def batch_generator(idata, target, batch_size, shuffle=True):
    nsamples = len(idata)
    perm = np.random.permutation(nsamples) if shuffle else np.arange(nsamples)
    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i + batch_size]
        yield idata[batch_idx], (target[batch_idx] if target is not None else None)

def load_preprocessed_dataset(prefix):
    token_vocab = Vocabulary()
    token_vocab.load(f'{prefix}.vocab')
    data = []
    for part in ['train', 'valid', 'test']:
        with np.load(f'{prefix}.{part}.npz') as set_data:
            idata, target = set_data['idata'], set_data['target']
            data.append((idata, target))
            print(f'Number of samples ({part}): {len(target)}')
    print(f'Vocabulary size: {len(token_vocab)}')
    return token_vocab, data

# %% Attention modules
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, bias=True):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        y, _ = scaled_dot_product_attention(q, k, v)
        y = self.out_proj(y)
        return y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        B, W, E = x.shape
        q = self.q_proj(x).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
        y, _ = scaled_dot_product_attention(q, k, v, dropout=self.attn_dropout)
        y = y.transpose(1, 2).contiguous().view(B, W, E)
        y = self.out_proj(y)
        return y

# %% Transformer layers
class TransformerLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiHeadTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# %% Model definitions

# Baseline: single TransformerLayer, single-head attention, sum pooling
class Baseline(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        self.att = TransformerLayer(embedding_dim)
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, x):
        e = self.emb(x)
        u = e + self.position_embedding
        v = self.att(u)
        x = v.sum(dim=1)
        return self.lin(x)

# Model 1: 2-layer stacked Transformer
class MultiLayerTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_layers=2,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, x):
        e = self.emb(x)
        h = e + self.position_embedding
        for layer in self.layers:
            h = layer(h)
        return self.lin(h.sum(dim=1))

# Model 2: Multi-head attention (4 heads)
class MultiHeadTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads=4,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        self.att = MultiHeadTransformerLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)

    def forward(self, x):
        e = self.emb(x)
        u = e + self.position_embedding
        v = self.att(u)
        return self.lin(v.sum(dim=1))

# Model 3: Shared input/output embeddings + mean pooling
class SharedEmbeddingTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads=4,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.att = MultiHeadTransformerLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)
        self.output_bias = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        e = self.emb(x)
        u = e + self.position_embedding
        v = self.att(u)
        x = v.mean(dim=1)
        return F.linear(x, self.emb.weight, self.output_bias)

# Model 4: MLP over concatenated embeddings
class MLPPredictor(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_dim=512,
                 dropout=0.2, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)
        input_dim = context_words * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_embeddings),
        )

    def forward(self, x):
        e = self.emb(x)
        u = e + self.position_embedding
        return self.mlp(u.view(u.size(0), -1))

# Model 5: Deep multi-head + shared embeddings + cosine LR
class DeepSharedTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.layers = nn.ModuleList([
            MultiHeadTransformerLayer(embedding_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        e = self.emb(x)
        h = e + self.position_embedding
        for layer in self.layers:
            h = layer(h)
        h = self.layer_norm(h)
        x = h.mean(dim=1)
        return F.linear(x, self.emb.weight, self.output_bias)

# Model 6: CLS token for prediction
class CLSTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads=4,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.position_embedding = nn.Parameter(
            torch.empty(1 + context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)
        self.att = MultiHeadTransformerLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)

    def forward(self, x):
        B = x.size(0)
        e = self.emb(x)
        cls = self.cls_token.expand(B, -1, -1)
        u = torch.cat([cls, e], dim=1)
        u = u + self.position_embedding
        v = self.att(u)
        return self.lin(v[:, 0, :])

# Model 7: Pre-norm Transformer (Llama-style)
class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class PreNormTransformer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(embedding_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_bias = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        e = self.emb(x)
        h = e + self.position_embedding
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        x = h.mean(dim=1)
        return F.linear(x, self.emb.weight, self.output_bias)

# %% Training and evaluation functions
def train_epoch(model, criterion, optimizer, idata, target, batch_size, device, scheduler=None, log=False):
    model.train()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    for X, y in batch_generator(idata, target, batch_size, shuffle=True):
        X = torch.tensor(X, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        ncorrect += (torch.max(output, 1)[1] == y).sum().item()
        ntokens += y.numel()
        niterations += 1
        if log and (niterations == 200 or niterations == 500 or niterations % 1000 == 0):
            print(f'  step={niterations}, accuracy={100*ncorrect/ntokens:.1f}%, loss={total_loss/ntokens:.3f}')
    return 100.0 * ncorrect / ntokens, total_loss / ntokens

def evaluate(model, criterion, idata, target, batch_size, device):
    model.eval()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    y_pred = []
    with torch.no_grad():
        for X, y in batch_generator(idata, target, batch_size, shuffle=False):
            X = torch.tensor(X, dtype=torch.long, device=device)
            output = model(X)
            if target is not None:
                y = torch.tensor(y, dtype=torch.long, device=device)
                loss = criterion(output, y)
                total_loss += loss.item()
                ncorrect += (torch.max(output, 1)[1] == y).sum().item()
                ntokens += y.numel()
            else:
                y_pred.append(torch.max(output, 1)[1].cpu().numpy())
    if target is not None:
        return 100.0 * ncorrect / ntokens, total_loss / ntokens
    return np.concatenate(y_pred)

# %% Setup
pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print("WARNING: Training without GPU can be very slow!")

# %% Load data
vocab, data = load_preprocessed_dataset(params.preprocessed)

valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')
tokens = valid_x_df.columns[1:]
valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')
valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')

test_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
test_x = test_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

context_words = params.window_size - 1

# %% Define all model configurations
MODEL_CONFIGS = [
    ('Baseline (1L, 1H, sum)',
     lambda: Baseline(len(vocab), 256, context_words),
     {'epochs': 2, 'batch_size': 2048, 'lr': 1e-3, 'scheduler': False}),

    ('Model 2: Multi-Head (4H)',
     lambda: MultiHeadTransformer(len(vocab), 256, num_heads=4, context_words=context_words),
     {'epochs': 2, 'batch_size': 2048, 'lr': 1e-3, 'scheduler': False}),

    ('Model 3: Shared Emb + Mean Pool',
     lambda: SharedEmbeddingTransformer(len(vocab), 256, num_heads=4, context_words=context_words),
     {'epochs': 2, 'batch_size': 2048, 'lr': 1e-3, 'scheduler': False}),

    ('Model 6: CLS Token',
     lambda: CLSTransformer(len(vocab), 256, num_heads=4, context_words=context_words),
     {'epochs': 2, 'batch_size': 2048, 'lr': 1e-3, 'scheduler': False}),
]

# %% Train all models and collect results
criterion = nn.CrossEntropyLoss(reduction='sum')
results = []
saved_states = {}  # Save best state_dict per model to avoid retraining

for name, model_fn, hparams in MODEL_CONFIGS:
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')

    model = model_fn().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {num_params:,}')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    scheduler = None
    if hparams['scheduler']:
        total_steps = hparams['epochs'] * (len(data[0][0]) // hparams['batch_size'] + 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-6)

    t0 = time.time()
    best_ep_acc = 0
    best_state = None

    for epoch in range(hparams['epochs']):
        train_acc, train_loss = train_epoch(
            model, criterion, optimizer,
            data[0][0], data[0][1], hparams['batch_size'], device, scheduler, log=True)
        wiki_acc, wiki_loss = evaluate(model, criterion, data[1][0], data[1][1], hparams['batch_size'], device)
        ep_acc, ep_loss = evaluate(model, criterion, valid_x, valid_y, hparams['batch_size'], device)
        print(f'  Epoch {epoch}: train={train_acc:.1f}% | wiki_valid={wiki_acc:.1f}% | EP_valid={ep_acc:.1f}% (loss={ep_loss:.3f})')
        if ep_acc > best_ep_acc:
            best_ep_acc = ep_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0

    # Restore best checkpoint and evaluate
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    wiki_acc, wiki_loss = evaluate(model, criterion, data[1][0], data[1][1], hparams['batch_size'], device)
    ep_acc, ep_loss = evaluate(model, criterion, valid_x, valid_y, hparams['batch_size'], device)
    ppl = math.exp(ep_loss)

    # Save state for later reuse (on CPU to save GPU memory)
    saved_states[name] = best_state

    results.append({
        'Model': name,
        'Params': num_params,
        'Epochs': hparams['epochs'],
        'LR': hparams['lr'],
        'Time (s)': round(elapsed),
        'Wiki Valid Acc%': round(wiki_acc, 1),
        'EP Valid Acc%': round(ep_acc, 1),
        'EP Valid Loss': round(ep_loss, 3),
        'EP Perplexity': round(ppl, 1),
    })
    print(f'  >> Best: EP_acc={ep_acc:.1f}%, ppl={ppl:.1f}, time={elapsed:.0f}s')

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# %% Comparison table
print('\n' + '='*60)
print('  COMPARISON TABLE')
print('='*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# %% Generate submission with best model (no retraining needed)
best_idx = max(range(len(results)), key=lambda i: results[i]['EP Valid Acc%'])
best_name = results[best_idx]['Model']
print(f'\nBest model: {best_name} (EP Valid Acc = {results[best_idx]["EP Valid Acc%"]}%)')

# Reload best model from saved state
best_model_fn = MODEL_CONFIGS[best_idx][1]
model = best_model_fn().to(device)
model.load_state_dict({k: v.to(device) for k, v in saved_states[best_name].items()})
torch.save(model.state_dict(), params.modelname)

# %% Generate test predictions
y_pred = evaluate(model, criterion, test_x, None, params.batch_size, device)
y_token = [vocab.get_token(int(idx)) for idx in y_pred]

submission = pd.DataFrame({'id': test_x_df['id'], 'token': y_token}, columns=['id', 'token'])
print(submission.head())
submission.to_csv('submission.csv', index=False)
print('Submission saved to submission.csv')
