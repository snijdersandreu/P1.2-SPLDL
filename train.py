"""
Training and evaluation script for non-causal language modeling.
Trains all model variants and produces a comparison table.
"""

import os
import time
import pathlib
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from types import SimpleNamespace

from models import (
    Baseline,
    MultiLayerTransformer,
    MultiHeadTransformer,
    SharedEmbeddingTransformer,
    MLPPredictor,
    DeepSharedTransformer,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_VERSION = 'ca-100'
COMPETITION_ROOT = '../input/competitions/wordvectors-2026'
DATASET_ROOT = f'../input/notebooks/jarfo1/text-preprocessing/data/{DATASET_VERSION}'
WORKING_ROOT = f'data/{DATASET_VERSION}'
DATASET_PREFIX = 'ca.wiki'


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_preprocessed_dataset(prefix):
    token_vocab = Vocabulary()
    token_vocab.load(f'{prefix}.vocab')
    data = []
    for part in ['train', 'valid', 'test']:
        with np.load(f'{prefix}.{part}.npz') as set_data:
            idata, target = set_data['idata'], set_data['target']
            data.append((idata, target))
            print(f'  {part}: {len(target):,} samples')
    print(f'  Vocabulary size: {len(token_vocab):,}')
    return token_vocab, data


def batch_generator(idata, target, batch_size, shuffle=True):
    nsamples = len(idata)
    perm = np.random.permutation(nsamples) if shuffle else np.arange(nsamples)
    for i in range(0, nsamples, batch_size):
        batch_idx = perm[i:i + batch_size]
        yield idata[batch_idx], (target[batch_idx] if target is not None else None)


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_epoch(model, criterion, optimizer, idata, target, batch_size, device, scheduler=None):
    model.train()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
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
    accuracy = 100.0 * ncorrect / ntokens
    avg_loss = total_loss / ntokens
    return accuracy, avg_loss


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
                pred = torch.max(output, 1)[1].detach().cpu().numpy()
                y_pred.append(pred)
    if target is not None:
        return 100.0 * ncorrect / ntokens, total_loss / ntokens
    else:
        return np.concatenate(y_pred)


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

def get_model_configs(vocab_size, context_words=6):
    """Return a dict of model_name -> (constructor_fn, hyperparams_dict)."""
    configs = {
        'Baseline (1-layer, 1-head)': {
            'model_fn': lambda: Baseline(vocab_size, 256, context_words),
            'epochs': 4,
            'batch_size': 2048,
            'lr': 1e-3,
            'use_scheduler': False,
            'description': 'emb=256, ff=512, 1 layer, 1 head, sum pool',
        },
        'Model 1: 2-layer Transformer': {
            'model_fn': lambda: MultiLayerTransformer(
                vocab_size, 256, num_layers=2, dim_feedforward=512,
                dropout=0.1, context_words=context_words),
            'epochs': 4,
            'batch_size': 2048,
            'lr': 1e-3,
            'use_scheduler': False,
            'description': 'emb=256, ff=512, 2 layers, 1 head, sum pool',
        },
        'Model 2: Multi-head attention (4 heads)': {
            'model_fn': lambda: MultiHeadTransformer(
                vocab_size, 256, num_heads=4, dim_feedforward=512,
                dropout=0.1, context_words=context_words),
            'epochs': 4,
            'batch_size': 2048,
            'lr': 1e-3,
            'use_scheduler': False,
            'description': 'emb=256, ff=512, 1 layer, 4 heads, sum pool',
        },
        'Model 3: Shared embeddings + mean pool': {
            'model_fn': lambda: SharedEmbeddingTransformer(
                vocab_size, 256, num_heads=4, dim_feedforward=512,
                dropout=0.1, context_words=context_words),
            'epochs': 4,
            'batch_size': 2048,
            'lr': 1e-3,
            'use_scheduler': False,
            'description': 'emb=256, ff=512, 1 layer, 4 heads, mean pool, tied weights',
        },
        'Model 4: MLP (no attention)': {
            'model_fn': lambda: MLPPredictor(
                vocab_size, 256, hidden_dim=512,
                dropout=0.2, context_words=context_words),
            'epochs': 4,
            'batch_size': 2048,
            'lr': 1e-3,
            'use_scheduler': False,
            'description': 'emb=256, hidden=512, 2 hidden layers, concat+MLP',
        },
        'Model 5: Deep MH + shared emb + scheduler': {
            'model_fn': lambda: DeepSharedTransformer(
                vocab_size, 256, num_heads=4, num_layers=2,
                dim_feedforward=512, dropout=0.1, context_words=context_words),
            'epochs': 5,
            'batch_size': 2048,
            'lr': 5e-4,
            'use_scheduler': True,
            'description': 'emb=256, ff=512, 2 layers, 4 heads, mean pool, tied, cosine LR',
        },
    }
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pathlib.Path(WORKING_ROOT).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: Training without GPU can be very slow!")

    print("Loading dataset...")
    prefix = f'{DATASET_ROOT}/{DATASET_PREFIX}'
    vocab, data = load_preprocessed_dataset(prefix)
    train_data, wiki_valid_data, wiki_test_data = data

    # Competition validation set
    valid_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_valid.csv')
    tokens = valid_x_df.columns[1:]
    valid_x = valid_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')
    valid_y_df = pd.read_csv(f'{COMPETITION_ROOT}/y_valid.csv')
    valid_y = valid_y_df['token'].apply(vocab.get_index).to_numpy(dtype='int32')

    # Competition test set
    test_x_df = pd.read_csv(f'{COMPETITION_ROOT}/x_test.csv')
    test_x = test_x_df[tokens].apply(vocab.get_index).to_numpy(dtype='int32')

    context_words = train_data[0].shape[1]  # window_size - 1
    configs = get_model_configs(len(vocab), context_words)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    results = []

    for name, cfg in configs.items():
        print(f'\n{"="*70}')
        print(f'Training: {name}')
        print(f'  {cfg["description"]}')
        print(f'{"="*70}')

        model = cfg['model_fn']().to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'  Parameters: {num_params:,}')

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

        scheduler = None
        if cfg['use_scheduler']:
            total_steps = cfg['epochs'] * (len(train_data[0]) // cfg['batch_size'] + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=1e-6)

        start_time = time.time()
        best_valid_acc = 0
        for epoch in range(cfg['epochs']):
            train_acc, train_loss = train_epoch(
                model, criterion, optimizer,
                train_data[0], train_data[1],
                cfg['batch_size'], device, scheduler)
            wiki_acc, wiki_loss = evaluate(
                model, criterion,
                wiki_valid_data[0], wiki_valid_data[1],
                cfg['batch_size'], device)
            ep_acc, ep_loss = evaluate(
                model, criterion, valid_x, valid_y,
                cfg['batch_size'], device)
            print(f'  Epoch {epoch}: train_acc={train_acc:.1f}% train_loss={train_loss:.3f} | '
                  f'wiki_valid_acc={wiki_acc:.1f}% | EP_valid_acc={ep_acc:.1f}% EP_loss={ep_loss:.3f}')
            if ep_acc > best_valid_acc:
                best_valid_acc = ep_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - start_time

        # Restore best model and evaluate on test set
        model.load_state_dict(best_state)
        wiki_acc, wiki_loss = evaluate(
            model, criterion,
            wiki_valid_data[0], wiki_valid_data[1],
            cfg['batch_size'], device)
        ep_acc, ep_loss = evaluate(
            model, criterion, valid_x, valid_y,
            cfg['batch_size'], device)

        # Generate test predictions
        y_pred = evaluate(model, criterion, test_x, None, cfg['batch_size'], device)
        y_tokens = [vocab.get_token(int(idx)) for idx in y_pred]

        # Save submission
        safe_name = name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')
        submission = pd.DataFrame({'id': test_x_df['id'], 'token': y_tokens})
        sub_path = f'{WORKING_ROOT}/submission_{safe_name}.csv'
        submission.to_csv(sub_path, index=False)

        # Save model
        model_path = f'{WORKING_ROOT}/{safe_name}.pt'
        torch.save(model.state_dict(), model_path)

        perplexity = math.exp(ep_loss)
        results.append({
            'Model': name,
            'Params': f'{num_params:,}',
            'Epochs': cfg['epochs'],
            'LR': cfg['lr'],
            'Train Time (s)': f'{elapsed:.0f}',
            'Wiki Valid Acc (%)': f'{wiki_acc:.1f}',
            'EP Valid Acc (%)': f'{ep_acc:.1f}',
            'EP Valid Loss': f'{ep_loss:.3f}',
            'EP Perplexity': f'{perplexity:.1f}',
            'Key Changes': cfg['description'],
        })
        print(f'  Best EP Valid: acc={ep_acc:.1f}%, loss={ep_loss:.3f}, ppl={perplexity:.1f}')
        print(f'  Training time: {elapsed:.0f}s')

    # Print comparison table
    print(f'\n{"="*70}')
    print('COMPARISON TABLE')
    print(f'{"="*70}')
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    df.to_csv(f'{WORKING_ROOT}/comparison_results.csv', index=False)
    print(f'\nResults saved to {WORKING_ROOT}/comparison_results.csv')


if __name__ == '__main__':
    main()
