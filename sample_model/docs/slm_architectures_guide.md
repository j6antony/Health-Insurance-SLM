# Comprehensive Guide to Small Language Model (SLM) Architectures

Small Language Models (SLMs)—typically ranging from **1M to 7B parameters**—have become one of the most active research and engineering areas in modern AI. By pairing targeted datasets with efficient architectural choices, SLMs can achieve remarkable performance on code generation, domain-specific Q&A, and reasoning while running on consumer GPUs, CPUs, or edge devices.

This guide provides a comprehensive overview of all major architecture families used to build SLMs today.

---

## Table of Contents

1. [Decoder-Only Autoregressive Transformers](#1-decoder-only-autoregressive-transformers)
   - 1.1 Modern LLaMA / Mistral / Gemma Style (SOTA Benchmark)
   - 1.2 Classic GPT-2 / GPT-3 Style
   - 1.3 Microsoft Phi Style (Deep & Focused)
2. [State-Space Models (SSMs) & Linear Architectures](#2-state-space-models-ssms--linear-architectures)
   - 2.1 Mamba & Mamba-2 (Selective State-Space)
   - 2.2 RWKV (Reinventing RNNs for the Transformer Era)
3. [Encoder-Decoder Architectures](#3-encoder-decoder-architectures)
   - 3.1 T5 / FLAN-T5 Architecture
4. [Mixture of Experts (MoE) SLMs](#4-mixture-of-experts-moe-slms)
   - 4.1 Sparse Router + Expert MLPs
5. [Hybrid Architectures](#5-hybrid-architectures)
   - 5.1 Hybrid Mamba-Attention (Jamba Style)
6. [Key Architectural Sub-Components Comparison](#6-key-architectural-sub-components-comparison)
7. [Choosing the Right Architecture for Your SLM](#7-choosing-the-right-architecture-for-your-slm)

---

## 1. Decoder-Only Autoregressive Transformers

Decoder-only architectures predict the next token given all previous tokens ($P(x_t \mid x_{<t})$). They dominate modern text and code generation.

### 1.1 Modern LLaMA / Mistral / Gemma Style (State of the Art)

Used in nearly all modern open-weight models (Llama 3, Qwen 2.5, Gemma 2, Mistral).

* **Positional Embeddings:** **RoPE (Rotary Position Embedding)** — encodes relative position directly into query and key vectors via rotation in the complex plane.
* **Normalization:** **RMSNorm (Root Mean Square Normalization)** placed **pre-layer** (before Attention and MLP). RMSNorm omits mean-centering to save computation while maintaining scale stability.
* **Feed-Forward Activation:** **SwiGLU (Swish-Gated Linear Unit)** — uses three weight matrices ($W_{gate}, W_{up}, W_{down}$) instead of two, significantly enhancing capacity for code and logic.
* **Attention mechanism:** Multi-Head Attention (MHA), Grouped-Query Attention (GQA) for larger models, or Multi-Query Attention (MQA) for memory savings.
* **Best for:** General text generation, Python code completion, multi-turn chat, instruction following.

```
       [ Input Tokens ]
              │
      [ Token Embedding ]
              │
   ┌──────────┴──────────┐
   │   RMSNorm           │
   │   RoPE Attention    │  ◄─ Causal Mask
   │   Residual Add      │
   ├─────────────────────┤
   │   RMSNorm           │
   │   SwiGLU FFN        │
   │   Residual Add      │
   └──────────┬──────────┘
              │ (Repeat N Layers)
      [ Final RMSNorm ]
              │
      [ Output Logits ]
```

### 1.2 Classic GPT-2 / GPT-3 Style

The architectural foundation popularized by OpenAI and Andrej Karpathy's `nanoGPT`.

* **Positional Embeddings:** **Learned Absolute Positional Embeddings** ($W_{pe} \in \mathbb{R}^{T \times d}$).
* **Normalization:** Standard **LayerNorm** with learnable gain ($\gamma$) and bias ($\beta$).
* **Activation:** **GELU (Gaussian Error Liquid Unit)** or GELU-tanh.
* **Feed-Forward Network:** 2-layer linear projections with expansion factor 4 ($d_{model} \to 4d_{model} \to d_{model}$).
* **Best for:** Simple educational implementations, baseline benchmarking, lightweight models.

### 1.3 Microsoft Phi Style (Deep & Focused)

Featured in Microsoft's Phi-1, Phi-2, and Phi-3 models.

* **Design Philosophy:** "Textbooks Are All You Need" — combining specialized synthetic datasets with deep, narrow transformer layers.
* **Key Traits:** Uses partial RoPE, fused FlashAttention operations, and aggressive depth-to-width ratios.
* **Best for:** High reasoning and code execution benchmarks at small parameter footprints (1.3B–3.8B parameters).

---

## 2. State-Space Models (SSMs) & Linear Architectures

Traditional transformers have $O(N^2)$ time and memory complexity with respect to context length $N$. SSMs achieve **$O(N)$ linear time complexity** and **constant $O(1)$ inference memory** per token.

### 2.1 Mamba & Mamba-2 (Selective State-Space)

Replaces self-attention with a continuous-time state-space dynamics matrix $(A, B, C)$ made input-dependent (selective).

* **Mechanism:** Tensors filter context selectively based on the current token, compressing context into a fixed-size hidden state.
* **Advantages:**
  - Infinitely long context inference without quadratic slowdown.
  - Zero KV-cache GPU memory bottleneck during generation.
* **Trade-off:** Slightly weaker than Self-Attention on needle-in-a-haystack exact retrieval tasks unless combined with attention.

### 2.2 RWKV (Reinventing RNNs for Transformers)

Combines the parallelizable training of Transformers with the fast $O(1)$ inference of RNNs.

* **Mechanism:** Uses a Time-Mixing and Channel-Mixing block operating on vector states.
* **Advantages:** Extremely lightweight inference footprint; easy to deploy on microcontroller / edge devices.

---

## 3. Encoder-Decoder Architectures

### 3.1 T5 / FLAN-T5 Architecture

Employs a bidirectional encoder to process the input prompt, followed by a causal decoder to generate the target sequence.

* **Best for:** Translation, code refactoring (input code $\to$ output code), summarization, text classification.
* **Trade-off:** Twice the forward-pass compute during prompt processing compared to decoder-only models.

---

## 4. Mixture of Experts (MoE) SLMs

MoE architectures replace the standard dense Feed-Forward Network (FFN) with multiple parallel "expert" networks.

* **Mechanism:** A top-$k$ Router (gating network) evaluates each token and routes it to only 1 or 2 experts out of $E$ total experts.
* **Example (e.g., 4x1M SLM):** Total parameter count is 4M, but compute per token is only equivalent to ~1.5M parameters.
* **Advantages:** High knowledge capacity without proportional compute cost.
* **Trade-off:** Higher VRAM memory footprint (all expert weights must be loaded in RAM/VRAM).

---

## 5. Hybrid Architectures

### 5.1 Hybrid Mamba-Attention (e.g., Jamba)

Interleaves SSM (Mamba) layers with occasional Self-Attention layers (e.g. 1 Attention layer for every 4 Mamba layers).

* **Advantages:** Combines the fast linear context processing of Mamba with the exact retrieval precision of Self-Attention.

---

## 6. Key Architectural Sub-Components Comparison


| Feature                 | Classic (GPT-2)               | Modern (LLaMA / Mistral)                 | State-Space (Mamba)          |
| ------------------------- | ------------------------------- | ------------------------------------------ | ------------------------------ |
| **Attention Mechanism** | Multi-Head Self-Attention     | Multi-Head / Grouped-Query Attention     | Selective SSM (No Attention) |
| **Positional Encoding** | Absolute Learned ($W_{pe}$)   | RoPE (Rotary Position Embeddings)        | Implicit in State Dynamics   |
| **Normalization**       | LayerNorm                     | Pre-RMSNorm                              | RMSNorm                      |
| **FFN Activation**      | GELU                          | SwiGLU                                   | Gated Conv + SILU            |
| **Time Complexity**     | $O(N^2)$                      | $O(N^2)$ (or $O(N)$ with FlashAttention) | $O(N)$ Linear                |
| **Inference Memory**    | Grows with Context (KV-Cache) | Grows with Context (GQA reduces)         | Constant$O(1)$ Memory        |

---

## 7. Choosing the Right Architecture for Your SLM

* **For Learning Modern PyTorch & Building a Flexible SLM:** Choose **Modern LLaMA-Style Architecture** (RoPE + RMSNorm + SwiGLU + BPE).
* **For Ultra-Fast Edge / Mobile Deployment:** Choose **Mamba / RWKV** or **Quantized LLaMA Micro**.
* **For Code Transformation (Input Code $\to$ Fixed Output Code):** Choose **T5 Encoder-Decoder**.
