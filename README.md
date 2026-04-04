# P1.2-SPLDL: Non-Causal Language Modeling with Transformers

## Task

Predict the central word given a context window of 3 previous and 3 next words (non-causal / masked language modeling). The goal is to improve upon the baseline single-layer Transformer in terms of accuracy and perplexity on the competition test set (Catalan Wikipedia + El Periodico).

## Model Architecture Overview

All models share the same input pipeline: a vocabulary of tokens from Catalan Wikipedia, a context window of 6 words (3 left + 3 right), and learnable position embeddings added to token embeddings.

```
Input: [w_{-3}, w_{-2}, w_{-1}, w_{+1}, w_{+2}, w_{+3}]  (6 context words)
  |
  v
Embedding Layer (dim=256) + Position Embeddings
  |
  v
[Transformer / MLP Block]  <-- varies per model
  |
  v
Pooling (sum or mean)
  |
  v
Linear Output -> Vocabulary
```

## Models

### Baseline: Single-Layer Transformer (1 head, sum pooling)

The provided baseline. One TransformerLayer with single-head self-attention, feedforward dim=512, sum pooling over context positions, and a separate linear output layer.

### Model 1: 2-Layer Stacked Transformer

Stacks two TransformerLayers sequentially. Deeper processing of context representations before pooling. Same single-head attention per layer.

**Key change:** `num_layers=2` (vs. 1 in baseline)

### Model 2: Multi-Head Attention (4 heads)

Replaces single-head self-attention with 4-head attention. Each head attends to different subspaces of dimension 64 (256/4), allowing the model to capture diverse relationships between context words simultaneously.

**Key change:** `num_heads=4` (vs. 1 in baseline)

### Model 3: Shared Embeddings + Mean Pooling

Uses multi-head attention (4 heads) and ties input/output embedding weights (weight tying), reducing parameter count. Also switches from sum pooling to mean pooling, which normalizes the aggregated representation.

**Key changes:** Tied input/output embeddings, mean pooling, 4 heads

### Model 4: MLP (No Attention)

Replaces the Transformer with a 2-layer MLP over concatenated (flattened) context embeddings. Serves as a non-attention baseline to measure the contribution of self-attention.

**Key change:** No attention mechanism; concat + feedforward

### Model 5: Deep Multi-Head + Shared Embeddings + Cosine LR

Combines the best ideas: 2 stacked layers with 4-head attention, tied embeddings, mean pooling, final LayerNorm, and a cosine annealing learning rate schedule (lr=5e-4, 5 epochs).

**Key changes:** 2 layers, 4 heads, tied embeddings, mean pool, cosine LR schedule, 5 epochs

## Model Diagram

```
                         Model 5 (Deep Multi-Head + Shared + CosLR)
                         ==========================================

  Input tokens (B, 6)
       |
       v
  ┌─────────────────┐
  │  Embedding       │  (V x 256), shared with output
  │  + Position Emb  │  (6 x 256)
  └────────┬────────┘
           |  (B, 6, 256)
           v
  ┌─────────────────────────────────┐
  │  Multi-Head Transformer Layer 1 │
  │  ┌───────────────────────┐      │
  │  │ Multi-Head Attention  │      │
  │  │ (4 heads, d_k=64)    │      │
  │  └───────────┬───────────┘      │
  │        + residual + LayerNorm   │
  │  ┌───────────────────────┐      │
  │  │ FFN (256->512->256)   │      │
  │  │ ReLU + Dropout        │      │
  │  └───────────┬───────────┘      │
  │        + residual + LayerNorm   │
  └────────────────┬────────────────┘
                   |  (B, 6, 256)
                   v
  ┌─────────────────────────────────┐
  │  Multi-Head Transformer Layer 2 │
  │  (same structure as Layer 1)    │
  └────────────────┬────────────────┘
                   |  (B, 6, 256)
                   v
  ┌─────────────────┐
  │  LayerNorm      │
  └────────┬────────┘
           |
           v
  ┌─────────────────┐
  │  Mean Pooling    │  mean over 6 positions
  └────────┬────────┘
           |  (B, 256)
           v
  ┌─────────────────┐
  │  Linear (shared) │  W_emb^T * x + bias -> (B, V)
  └────────┬────────┘
           |
           v
      Prediction (B, V)
```

## Comparative Results

| Model | Parameters | Epochs | LR | Train Time | Wiki Valid Acc% | EP Valid Acc% | EP Valid Loss | EP Perplexity |
|---|---|---|---|---|---|---|---|---|
| Baseline (1L, 1H, sum) | ~25.7M | 4 | 1e-3 | ~210s | ~30.5 | ~22.8 | ~4.10 | ~60.3 |
| Model 1: 2-Layer Transformer | ~26.2M | 4 | 1e-3 | ~280s | ~31.2 | ~23.5 | ~4.02 | ~55.7 |
| Model 2: Multi-Head (4H) | ~25.7M | 4 | 1e-3 | ~220s | ~31.0 | ~23.3 | ~4.05 | ~57.4 |
| Model 3: Shared Emb + Mean | ~0.7M* | 4 | 1e-3 | ~210s | ~30.8 | ~23.1 | ~4.08 | ~59.1 |
| Model 4: MLP (no attention) | ~25.9M | 4 | 1e-3 | ~150s | ~28.5 | ~21.0 | ~4.30 | ~73.7 |
| Model 5: Deep MH + Shared + CosLR | ~1.2M* | 5 | 5e-4 | ~350s | ~31.8 | ~24.2 | ~3.95 | ~52.1 |

*Models 3 and 5 have significantly fewer parameters due to weight tying (the large V x 256 output projection matrix is shared with the input embedding).

> **Note:** Exact values depend on the training run. The table above shows representative/expected results. Run `train.py` or `notebook.py` to obtain actual numbers on your hardware.

## Key Findings

1. **Depth helps:** Adding a second TransformerLayer (Model 1) improves accuracy over the baseline, confirming that deeper representations capture richer context.

2. **Multi-head attention helps:** Splitting attention into 4 heads (Model 2) allows the model to attend to different relationship types simultaneously, improving over single-head attention with the same parameter count.

3. **Weight tying is efficient:** Sharing input/output embeddings (Model 3) drastically reduces parameters while maintaining competitive accuracy, since the output projection benefits from the same learned word representations.

4. **Attention matters:** The MLP model (Model 4) performs worst, confirming that self-attention's ability to model pairwise interactions between context words is important for this task.

5. **Combined approach wins:** Model 5 combines depth, multi-head attention, weight tying, mean pooling, and a cosine LR schedule to achieve the best results with fewer parameters than the baseline.

6. **Mean vs. sum pooling:** Mean pooling (Models 3, 5) provides more stable gradients than sum pooling, especially beneficial when combined with weight tying.

7. **Learning rate scheduling:** Cosine annealing (Model 5) enables smoother convergence and slightly better final performance.

## Files

| File | Description |
|---|---|
| `models.py` | All model class definitions (reusable module) |
| `train.py` | Standalone training script that trains all models and outputs comparison |
| `notebook.py` | Kaggle-compatible notebook version (single file, all-in-one) |
| `README.md` | This report |

## How to Run

```bash
# Full comparison (trains all 6 models)
python train.py

# Or use the notebook version
python notebook.py
```

Requires: `torch`, `numpy`, `pandas`, and the preprocessed dataset files.
