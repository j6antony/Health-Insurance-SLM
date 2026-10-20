# Module 4: How Training & Text Generation Actually Work

---

## 1. How Training Works: "Next-Token Prediction"

A language model is essentially a very smart **auto-complete engine**.

Given the prefix:
> `"Once upon a"`

The model must predict the probability distribution for the 4th word:
* `"time"` $\rightarrow 85\%$
* `"hill"` $\rightarrow 5\%$
* `"banana"` $\rightarrow 0.001\%$

### The Math: Cross-Entropy Loss
We take the predicted probabilities for the correct target token and compute:
$$\mathcal{L} = -\log P(\text{target\_token})$$

If the model is completely guessing randomly at the beginning of training (across 4,096 tokens):
$$\mathcal{L}_{\text{initial}} \approx \ln(4096) \approx \mathbf{8.318}$$

Notice that in our smoke test log, our initial loss was **8.358**! That mathematically proves our model started from a clean, unbiased random initialization.
As training progresses, the loss drops down towards **2.0 - 1.5**, meaning the model has learned English grammar and vocabulary!

---

## 2. Training Hardware Optimizations (For Our 4GB VRAM GPU)

How did we make this run fast on an RTX 3050 Laptop GPU?

1. **Automatic Mixed Precision (FP16)**:
   * Standard 32-bit floats take 4 bytes each.
   * FP16 takes 2 bytes each $\rightarrow$ **Cuts memory usage in half** and utilizes the Tensor Cores of the RTX 3050.
2. **AdamW Optimizer**:
   * Uses momentum and adaptive learning rates, with decoupled weight decay ($0.05$) to prevent weights from blowing up.
3. **Cosine Annealing Learning Rate with Warmup**:
   * **Warmup**: Starts at a tiny learning rate ($8 \times 10^{-6}$) and ramps up over the first 200 steps so the model doesn't get shocked by huge gradients initially.
   * **Cosine Decay**: Gradually slows down the learning rate following a smooth cosine curve as the model approaches its best performance.
4. **Gradient Clipping**:
   * Caps the gradient norm at $1.0$ so no single crazy batch can derail training.

---

## 3. How Story Generation Works (Inference)

Once the model is trained, how does it write a story?

We pass a starting prompt: `"Lily found a little"`
1. The model predicts probabilities for the next token.
2. We **sample** a token using 3 controls:
   * **Temperature ($T$)**:
     * $T = 0$: Greedy (picks the #1 highest probability token every time; can become robotic or repetitive).
     * $T = 0.7 - 0.9$: Balanced creativity and coherence.
     * $T > 1.5$: Wild and chaotic.
   * **Top-K**: Only considers the top $K$ (e.g., 40) most likely words; ignores all absurd words.
   * **Top-P (Nucleus Sampling)**: Dynamically selects the smallest group of words whose combined probability reaches $P$ (e.g., $90\%$).
3. Append the chosen word to the prompt, and feed it back into the model to predict the next word!
4. Repeat until it emits `<|endoftext|>` or reaches `max_new_tokens`.

---

## 4. Summary Checklist for Your Presentation

- [x] **Data**: 50,000 synthetic children's stories (`TinyStories`).
- [x] **Tokenizer**: 4,096 custom Byte-Level BPE (No `<unk>`, low memory overhead).
- [x] **Architecture**: 2.56M parameter Micro-LLaMA (RMSNorm, RoPE, SwiGLU, Weight Tying).
- [x] **Training**: Mixed-precision FP16, AdamW, Cosine schedule with warmup.
- [x] **Generation**: Autoregressive sampling with Temperature, Top-K, and Top-P.
