# TinyStories Micro-LLaMA (SLM) Architecture Specification

This document details the architectural blueprint for building an ultra-compact Small Language Model (SLM) from scratch using PyTorch and training it on the `roneneldan/TinyStories` dataset.

---

## 1. System Overview

```
                      +-----------------------------+
                      |   TinyStories Dataset       |
                      |   (HuggingFace / Streaming) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Custom BPE Tokenizer        |
                      | (Vocab: 4,096 tokens)       |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Micro-LLaMA Architecture    |
                      |   - RMSNorm                 |
                      |   - Rotary Embeddings (RoPE)|
                      |   - SwiGLU Feed-Forward     |
                      |   - Scaled Dot-Product Attn |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | PyTorch Training Pipeline   |
                      |   - Mixed Precision (FP16)  |
                      |   - Cosine Decay + Warmup   |
                      |   - RTX 3050 Optimized      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Interactive Generation / CLI|
                      |   - Top-p / Top-k Sampling  |
                      |   - Temperature Control     |
                      +-----------------------------+
```

---

## 2. Model Hyperparameters (~3M - 5M Parameters)

Designed specifically for swift iteration, pure from-scratch understanding, and fitting comfortably in 4GB VRAM:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | 4,096 | Compact BPE vocabulary customized to TinyStories |
| `max_seq_len` | 256 | Context window (TinyStories paragraphs typically ~100-200 words) |
| `dim` ($d_{model}$) | 192 | Hidden dimension |
| `n_layers` | 4 | Number of Transformer decoder layers |
| `n_heads` | 6 | Multi-head attention heads ($d_{head} = 32$) |
| `n_kv_heads` | 6 | Grouped-Query / Multi-Head Attention |
| `hidden_dim` (FFN) | 512 | SwiGLU expansion dimension ($\approx \frac{8}{3} \times dim$) |
| `norm_eps` | $10^{-5}$ | RMSNorm epsilon stability |
| `dropout` | 0.05 | Regularization |

---

## 3. Core Architectural Components

### 3.1 RMSNorm (Root Mean Square Layer Normalization)
Replaces standard LayerNorm by removing the mean-centering step:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

### 3.2 Rotary Positional Embeddings (RoPE)
Encodes positional information directly into queries and keys via complex rotation.

### 3.3 SwiGLU Feed-Forward Network
Replaces standard ReLU/GELU MLP with Gated Linear Unit using SiLU (Swish):
$$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$

---

## 4. Pipeline Modules

1. `src/tokenizer/train_tinystories_tokenizer.py`:
   - Downloads a representative subset of TinyStories.
   - Trains a Byte-Level BPE tokenizer with 4,096 tokens.
   - Saves `data/tinystories_tokenizer.json`.

2. `src/model/llama.py`:
   - From-scratch implementation of `RMSNorm`, `RoPE`, `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, and `MicroLlamaForCausalLM`.
   - Standalone, zero external model dependencies (pure PyTorch).

3. `src/data/dataset.py`:
   - Tokenizes text chunks and prepares efficient causal autoregressive batches (`input_ids` and `labels`).

4. `src/train.py`:
   - Training loop with AdamW, Cosine Annealing learning rate schedule, linear warmup.
   - Mixed-precision (`torch.amp.autocast`) for fast throughput on RTX 3050.
   - Periodic sample generation to watch stories emerge in real-time.

5. `src/generate.py`:
   - Inference CLI with temperature, top-k, and top-p (nucleus) sampling.
