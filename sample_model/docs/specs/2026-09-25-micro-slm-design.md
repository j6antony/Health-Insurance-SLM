# Specification: Micro Python SLM (Small Language Model from Scratch)

**Date:** 2026-09-25  
**Topic:** Micro SLM from First Principles with PyTorch  
**Goal:** Build and understand a ~2.1M parameter Small Language Model trained on Python code, with full step-by-step interactive learning.

---

## 1. Project Philosophy & Learning Objectives
* **First Principles Understanding:** Every component (Tokenizer, Embeddings, Attention, RoPE, RMSNorm, SwiGLU, Loss, Sampling) will be built and explained step-by-step.
* **Modern LLM Architecture:** Uses the same architectural blueprint as modern state-of-the-art models (LLaMA 3, Mistral, Gemma).
* **Self-Contained & Fast:** Designed to train in 3–5 minutes on standard hardware so feedback is instant.

---

## 2. Model Architecture & Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Architecture Type** | Causal Decoder-Only Transformer | Standard generative language modeling |
| **Vocabulary Size ($V$)** | 2,048 tokens | Compact BPE vocabulary tuned for Python syntax |
| **Context Length ($T_{max}$)** | 256 tokens | Sufficient context for complete Python functions |
| **Model Dimension ($d_{model}$)** | 128 | Vector representation width per token |
| **Layers ($N_{layers}$)** | 6 Transformer Blocks | Depth for syntactic and semantic feature extraction |
| **Attention Heads ($N_{heads}$)** | 4 heads | Multi-aspect attention ($d_{head} = 32$) |
| **FFN Dimension ($d_{ffn}$)** | 344 | SwiGLU expansion ($\approx \frac{8}{3} d_{model}$) |
| **Positional Encoding** | RoPE (Rotary Position Embeddings) | Modern relative positional representation |
| **Normalization** | Pre-RMSNorm ($\epsilon = 1e-5$) | Fast, stable scale-invariant normalization |
| **Feed-Forward Activation** | SwiGLU | Gated linear unit for superior capacity |
| **Total Parameters** | ~2,145,000 (~2.1M) | Ultra-lightweight and fast to train |

---

## 3. The 6-Step Implementation & Learning Roadmap

### Step 1: Environment & Tokenizer from Scratch
* Understand how text becomes numbers: Characters vs Words vs Sub-word Tokens (BPE).
* Build and train a custom Byte-Pair Encoding (BPE) tokenizer aware of Python indentation and syntax.
* Test encoding, decoding, and special tokens (`<s>`, `</s>`, `<pad>`).

### Step 2: Tensor Embeddings & Rotary Position Embeddings (RoPE)
* Understand token lookup embeddings ($W_E \in \mathbb{R}^{V \times d_{model}}$).
* Learn the math behind RoPE: why rotating query and key vectors in 2D pairs encodes relative token distance without static positional lookup tables.

### Step 3: Multi-Head Causal Self-Attention with RoPE
* Understand Queries ($Q$), Keys ($K$), and Values ($V$).
* Implement the Causal Attention Mask (preventing tokens from seeing the future).
* Implement multi-head splitting, scaled dot-product attention, and output projection.

### Step 4: Normalization (RMSNorm) & Gated MLP (SwiGLU)
* Implement Root Mean Square Normalization (RMSNorm) vs LayerNorm.
* Implement the SwiGLU Feed-Forward Network ($W_{gate} \cdot \text{SiLU}(W_{up}) \cdot W_{down}$).
* Assemble the complete `TransformerBlock` and `MicroSLM` model class.

### Step 5: Data Pipeline, Loss Function & Training Loop
* Prepare the Python code dataset into chunked $(X, Y)$ autoregressive sequence pairs.
* Build the training loop with `AdamW`, learning rate warmup + cosine decay, and gradient clipping.
* Track training loss and validation loss curve.

### Step 6: Inference, Sampling & Interactive Code Generator
* Implement text generation with Temperature, Top-$k$, and Top-$p$ (Nucleus) sampling.
* Build an interactive CLI (`generate.py`) where you give a Python prompt and the model completes the code.
