# Module 3: Micro-LLaMA vs. GPT-2 Architecture (Why We Chose Modern Standards)

---

## 1. High-Level Comparison: The Evolution from 2019 to 2026

When building a language model, there are two common templates:
1. **The Old Template (GPT-2, 2019)**: Absolute Learned Positional Embeddings, LayerNorm, GELU MLP.
2. **The Modern Template (LLaMA / Mistral, 2023–2026)**: Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU MLP.

We chose the **Modern LLaMA architecture**. Here is why every single part was upgraded:

---

## 2. Component 1: Normalization (LayerNorm vs. RMSNorm)

### Why normalize?
Deep neural networks have signals passing through multiple layers. Without normalization, values can blow up (explode to infinity) or shrink to zero, crashing the training.

### How LayerNorm works (Old):
1. Calculates the mean $\mu = \frac{1}{d}\sum x_i$
2. Centers the values $(x_i - \mu)$
3. Calculates variance $\sigma^2$
4. Divides by standard deviation: $\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$

### How RMSNorm works (What WE Did):
Researchers discovered that **centering around the mean is unnecessary compute waste**! All the stability comes from scaling by the root-mean-square.
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$$

* **Why we did it**: Saves ~7% of GPU operations per layer while providing identical or better mathematical stability.

---

## 3. Component 2: Positional Encoding (Absolute vs. RoPE)

### The Problem:
Attention mechanisms look at all words simultaneously. If you scramble `"dog bites man"` and `"man bites dog"`, standard attention sees the exact same words! The model **must know the order of words**.

### Old Way (GPT-2 Absolute Embeddings):
Add a static learned vector for position 0, position 1, position 2...
* **Fatal Flaw**: Does not understand relative distances (it doesn't naturally know that token 5 is 2 steps away from token 3). It also completely breaks if you ask the model to generate past the trained sequence length.

### Modern Way: RoPE (Rotary Position Embeddings - What WE Did):
Instead of adding numbers, **we rotate the Query and Key vectors in 2D planes by an angle proportional to their position index**!

$$\mathbf{q}_m = \mathbf{R}_{\Theta, m} \mathbf{W}_q \mathbf{x}_m$$

* **Why we did it**:
  1. The dot product between word $m$ and word $n$ naturally depends on their **relative distance** $(m - n)$.
  2. Tokens close to each other attend more naturally, and long-range interactions decay smoothly.

---

## 4. Component 3: Feed-Forward Network (GELU vs. SwiGLU)

After attention gathers information from other words, the Feed-Forward Network (FFN) processes and thinks about what it learned.

### Old Way (GPT-2):
$$\text{Output} = \text{GELU}(x W_1) W_2$$
A standard two-layer Multi-Layer Perceptron (MLP).

### Modern Way: SwiGLU (What WE Did):
Uses a **Gated Linear Unit (GLU)** with the SiLU (Swish) activation function:
$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

One branch computes a gate ($x W_{\text{gate}}$), while the other branch computes values ($x W_{\text{up}}$). They are multiplied together.
* **Why we did it**: Extensive research across Meta, Google, and Mistral proved that SwiGLU converges significantly faster and produces lower validation loss for the exact same amount of compute.
