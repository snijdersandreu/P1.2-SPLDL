"""
Model definitions for non-causal language modeling.
Baseline: Single TransformerLayer with single-head self-attention.
Variants: Multi-layer, multi-head, shared embeddings, MLP, and combined.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot-Product Attention."""
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
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        y, _ = scaled_dot_product_attention(q, k, v)
        y = self.out_proj(y)
        return y


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention."""
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
        # Project and reshape to (B, num_heads, W, head_dim)
        q = self.q_proj(x).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
        y, _ = scaled_dot_product_attention(q, k, v, dropout=self.attn_dropout)
        # Concatenate heads: (B, W, E)
        y = y.transpose(1, 2).contiguous().view(B, W, E)
        y = self.out_proj(y)
        return y


# ---------------------------------------------------------------------------
# Transformer layer variants
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """Single transformer encoder layer with single-head attention."""
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
    """Transformer encoder layer with multi-head attention."""
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


# ---------------------------------------------------------------------------
# Model 0 (Baseline): Single TransformerLayer, single-head, sum pooling
# ---------------------------------------------------------------------------

class Baseline(nn.Module):
    """Baseline: single TransformerLayer with single-head attention."""
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
        y = self.lin(x)
        return y


# ---------------------------------------------------------------------------
# Model 1: 2-layer Transformer (stacked TransformerLayers)
# ---------------------------------------------------------------------------

class MultiLayerTransformer(nn.Module):
    """Stack multiple TransformerLayers for deeper representations."""
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
        u = e + self.position_embedding
        h = u
        for layer in self.layers:
            h = layer(h)
        x = h.sum(dim=1)
        y = self.lin(x)
        return y


# ---------------------------------------------------------------------------
# Model 2: Multi-head attention Transformer
# ---------------------------------------------------------------------------

class MultiHeadTransformer(nn.Module):
    """Single layer with multi-head attention (4 heads)."""
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
        x = v.sum(dim=1)
        y = self.lin(x)
        return y


# ---------------------------------------------------------------------------
# Model 3: Shared input/output embeddings + mean pooling
# ---------------------------------------------------------------------------

class SharedEmbeddingTransformer(nn.Module):
    """Tie input and output embedding weights (weight tying) and use mean pooling."""
    def __init__(self, num_embeddings, embedding_dim, num_heads=4,
                 dim_feedforward=512, dropout=0.1, context_words=6):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.att = MultiHeadTransformerLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        self.position_embedding = nn.Parameter(torch.empty(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)
        # Output projection shares weights with embedding
        self.output_bias = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        e = self.emb(x)
        u = e + self.position_embedding
        v = self.att(u)
        # Mean pooling instead of sum
        x = v.mean(dim=1)
        # Shared embedding: multiply by transposed embedding matrix
        y = F.linear(x, self.emb.weight, self.output_bias)
        return y


# ---------------------------------------------------------------------------
# Model 4: MLP over concatenated embeddings (no attention)
# ---------------------------------------------------------------------------

class MLPPredictor(nn.Module):
    """MLP baseline: concatenate context embeddings and pass through feedforward layers."""
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
        # Flatten: (B, W*E)
        flat = u.view(u.size(0), -1)
        y = self.mlp(flat)
        return y


# ---------------------------------------------------------------------------
# Model 5: Deep multi-head Transformer with shared embeddings and LR schedule
# (combined best practices)
# ---------------------------------------------------------------------------

class DeepSharedTransformer(nn.Module):
    """2-layer multi-head Transformer with shared embeddings and mean pooling."""
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
        # Shared output weights
        self.output_bias = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        e = self.emb(x)
        u = e + self.position_embedding
        h = u
        for layer in self.layers:
            h = layer(h)
        h = self.layer_norm(h)
        x = h.mean(dim=1)
        y = F.linear(x, self.emb.weight, self.output_bias)
        return y
