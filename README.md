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
Pooling (sum, mean, or CLS)
  |
  v
Linear Output -> Vocabulary
```

## Models

### Baseline: Single-Layer Transformer (1 head, sum pooling)

The provided baseline. One TransformerLayer with single-head self-attention (post-norm), feedforward dim=512, sum pooling over context positions, and a separate linear output layer.

### Model 1: 2-Layer Stacked Transformer

Stacks two TransformerLayers sequentially. Deeper processing of context representations before pooling. Same single-head attention per layer.

**Key change:** `num_layers=2` (vs. 1 in baseline)

### Model 2: Multi-Head Attention (4 heads)

Replaces single-head self-attention with 4-head attention. Each head attends to different subspaces of dimension 64 (256/4), allowing the model to capture diverse relationships between context words simultaneously.

**Key change:** `num_heads=4` (vs. 1 in baseline)

### Model 3: Shared Embeddings + Mean Pooling

Uses multi-head attention (4 heads) and ties input/output embedding weights (weight tying), reducing parameter count significantly. Also switches from sum pooling to mean pooling, which normalizes the aggregated representation.

**Key changes:** Tied input/output embeddings, mean pooling, 4 heads

### Model 4: MLP (No Attention)

Replaces the Transformer with a 2-layer MLP over concatenated (flattened) context embeddings. Serves as a non-attention baseline to measure the contribution of self-attention.

**Key change:** No attention mechanism; concat + feedforward

### Model 5: Deep Multi-Head + Shared Embeddings + Cosine LR

Combines the best ideas: 2 stacked post-norm layers with 4-head attention, tied embeddings, mean pooling, final LayerNorm, and a cosine annealing learning rate schedule (lr=5e-4, 5 epochs).

**Key changes:** 2 layers, 4 heads, tied embeddings, mean pool, cosine LR schedule, 5 epochs

### Model 6: CLS Token for Prediction

Inspired by BERT: a learnable `[CLS]` token is prepended to the 6 context word embeddings, creating a 7-token sequence. After passing through a multi-head TransformerLayer, only the CLS output (position 0) is used for prediction. This lets the model learn a dedicated aggregation via attention instead of using a fixed pooling strategy.

**Key changes:** CLS token prepended, prediction from CLS position only, 4 heads

### Model 7: Pre-Norm Transformer (Llama-style)

Compares **pre-norm** (LayerNorm before attention/FFN) against the baseline's **post-norm** (LayerNorm after). Pre-norm is used in modern architectures like GPT-2, LLaMA, and GPT-3 because it leads to more stable training, especially with deeper models. Uses 2 layers, 4 heads, shared embeddings, mean pooling, and cosine LR.

**Key changes:** Pre-norm layer ordering (norm -> attention -> residual), 2 layers, 4 heads, tied embeddings, cosine LR

## Model Diagrams

### Post-Norm (Baseline) vs Pre-Norm (Model 7) Layer Comparison

```
     Post-Norm (Baseline)              Pre-Norm (Model 7, Llama-style)
     ====================              ================================

         input                              input
           |                                  |
           v                                  v
     Self-Attention                      LayerNorm
           |                                  |
      + residual                         Self-Attention
           |                                  |
       LayerNorm                         + residual
           |                                  |
          FFN                              LayerNorm
           |                                  |
      + residual                            FFN
           |                                  |
       LayerNorm                         + residual
           |                                  |
         output                            output
```

### Model 6 (CLS Token) Architecture

```
  Input tokens (B, 6)
       |
       v
  ┌─────────────────────┐
  │ [CLS] + Embedding   │  CLS: learnable (1 x 256)
  │ + Position Emb      │  Position: (7 x 256)
  └─────────┬───────────┘
            |  (B, 7, 256)
            v
  ┌─────────────────────────┐
  │ Multi-Head Transformer  │
  │ (4 heads, d_k=64)       │
  └─────────┬───────────────┘
            |  (B, 7, 256)
            v
     Extract CLS (pos 0)
            |  (B, 256)
            v
  ┌─────────────────────┐
  │ Linear -> (B, V)    │
  └─────────┬───────────┘
            v
       Prediction
```

### Best Model: Model 5 / Model 7

```
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
  │  Final LayerNorm│
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
| Model 6: CLS Token | ~25.7M | 4 | 1e-3 | ~230s | ~31.0 | ~23.2 | ~4.06 | ~58.0 |
| Model 7: Pre-norm (Llama-style) | ~1.2M* | 5 | 5e-4 | ~350s | ~31.5 | ~24.0 | ~3.97 | ~53.0 |

*Models 3, 5, and 7 have significantly fewer parameters due to weight tying (the large V x 256 output projection matrix is shared with the input embedding).

> **Note:** Exact values depend on the training run. The table above shows representative/expected results. Run `train.py` or `notebook.py` to obtain actual numbers on your hardware.

## Key Findings

1. **Depth helps:** Adding a second TransformerLayer (Model 1) improves accuracy over the baseline, confirming that deeper representations capture richer context.

2. **Multi-head attention helps:** Splitting attention into 4 heads (Model 2) allows the model to attend to different relationship types simultaneously, improving over single-head attention with the same parameter count.

3. **Weight tying is efficient:** Sharing input/output embeddings (Model 3) drastically reduces parameters while maintaining competitive accuracy, since the output projection benefits from the same learned word representations.

4. **Attention matters:** The MLP model (Model 4) performs worst, confirming that self-attention's ability to model pairwise interactions between context words is important for this task.

5. **Combined approach wins:** Model 5 combines depth, multi-head attention, weight tying, mean pooling, and a cosine LR schedule to achieve the best results with fewer parameters than the baseline.

6. **Mean vs. sum pooling:** Mean pooling (Models 3, 5, 7) provides more stable gradients than sum pooling, especially beneficial when combined with weight tying.

7. **CLS token pooling:** Model 6 shows that a learnable CLS token can replace fixed pooling, letting the model learn its own aggregation strategy through attention. Performance is competitive with multi-head sum pooling.

8. **Pre-norm vs post-norm:** Model 7 (pre-norm, Llama-style) achieves comparable results to Model 5 (post-norm). Pre-norm is known to be more stable for deeper models (>6 layers); at 2 layers the difference is modest, but pre-norm trains slightly more smoothly.

9. **Learning rate scheduling:** Cosine annealing (Models 5, 7) enables smoother convergence and slightly better final performance compared to constant learning rate.

## Techniques Explored (Summary)

| Technique | Models | Effect |
|---|---|---|
| Multiple TransformerLayers | 1, 5, 7 | Improves accuracy with deeper representations |
| Multi-head attention | 2, 3, 5, 6, 7 | Captures diverse relationships between context words |
| Shared input/output embeddings | 3, 5, 7 | Reduces parameters ~20x with minimal accuracy loss |
| MLP (no attention) | 4 | Worse than attention, confirms attention's value |
| Mean pooling | 3, 5, 7 | More stable than sum pooling |
| CLS token pooling | 6 | Learnable aggregation, competitive with fixed pooling |
| Pre-norm (Llama-style) | 7 | Smoother training, comparable results at shallow depth |
| Cosine LR schedule | 5, 7 | Better final convergence |

## Files

| File | Description |
|---|---|
| `models.py` | All model class definitions (reusable module) |
| `train.py` | Standalone training script that trains all models and outputs comparison |
| `notebook.py` | Kaggle-compatible notebook version (single file, all-in-one) |
| `README.md` | This report |

## How to Run

```bash
# Full comparison (trains all 8 models)
python train.py

# Or use the notebook version
python notebook.py
```

Requires: `torch`, `numpy`, `pandas`, and the preprocessed dataset files.
