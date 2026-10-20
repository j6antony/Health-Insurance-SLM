# Team Presentation Guide: Building a Micro-LLaMA SLM on TinyStories

A comprehensive presentation and discussion guide explaining the architecture, design choices, trade-offs, and alternative approaches.

---

## 1. Executive Summary & Objective

- **Goal**: Build an ultra-compact Small Language Model (SLM) **completely from scratch in PyTorch** (~2.56M parameters), train its own custom tokenizer, and train the model on synthetic English children's stories (`roneneldan/TinyStories`).
- **Target Hardware**: Single consumer laptop GPU (NVIDIA RTX 3050 Laptop, 4GB VRAM).
- **Core Philosophy**: Prioritize modern architecture standards (LLaMA/Mistral 2024–2026 specs) over legacy Transformer designs (GPT-2 2019), keeping parameter count small enough to iterate rapidly while retaining state-of-the-art component designs.

---

## 2. End-to-End Workflow: What We Built

### Step 1: Data Acquisition & Preprocessing
- **Dataset**: `roneneldan/TinyStories` (Microsoft Research). A dataset of synthetically generated children's stories designed to prove that small models (<10M params) can learn grammar, reasoning, and consistent narratives if the vocabulary and domain complexity are bounded.
- **Corpus Caching**: Streamed 50,000 stories into `data/tinystories_corpus.txt` (~44 MB) separated by `<|endoftext|>` delimiters.

### Step 2: Custom Byte-Level BPE Tokenizer
- **Algorithm**: Byte-Level Byte-Pair Encoding (BPE) using Hugging Face's `tokenizers` library.
- **Vocabulary Size**: **4,096 tokens** (saved to `data/tinystories_tokenizer.json`).
- **Initial Alphabet**: Full 256-byte alphabet (guarantees zero out-of-vocabulary / `<unk>` tokens).
- **Compression**: Achieved ~4.22 characters per token on children's stories.

### Step 3: From-Scratch Model Architecture (Micro-LLaMA, 2.56M Params)
Implemented directly in PyTorch in `src/model/llama.py`:

| Layer / Feature | Implementation | Purpose |
| :--- | :--- | :--- |
| **Normalization** | `RMSNorm` | Replaces LayerNorm; omits mean-centering to save ~7% compute with equivalent stability. |
| **Position Embeddings** | **RoPE (Rotary Position Embeddings)** | Rotates Q and K in complex space. Enables relative distance decay and superior context extrapolation. |
| **Attention** | **Scaled Dot-Product Attention (SDPA)** | Uses `torch.nn.functional.scaled_dot_product_attention` for causal FlashAttention acceleration. |
| **Feed-Forward** | **SwiGLU** | Gated Linear Unit with SiLU activation: $\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$. |
| **Weight Tying** | $W_{\text{output}} = W_{\text{embed}}^T$ | Shares weights between token embeddings and the language model head to save 786,432 parameters. |

### Step 4: High-Throughput Training Pipeline (`src/train.py`)
- **Mixed Precision**: FP16 Automatic Mixed Precision (`torch.amp.autocast("cuda")` + `GradScaler`).
- **Optimizer**: AdamW with decoupled weight decay (applied only to 2D weight matrices; disabled for 1D biases/RMSNorm scales).
- **Learning Rate Schedule**: Cosine Annealing with linear warmup.
- **Gradient Clipping**: Capped at `max_norm=1.0` to avoid exploding gradients.
- **Auto-Evaluation**: Periodically prompts the model during training to sample live story generation.

### Step 5: Interactive Generation CLI (`src/generate.py`)
- Supports **Temperature scaling**, **Top-K filtering**, and **Top-P (nucleus) sampling**.

---

## 3. Deep Dive: What We Chose vs. What We Did NOT Do (And Why)

Use this section to showcase your engineering trade-offs during team discussions:

| Decision Area | What We Did | What We Did NOT Do | Why We Made This Decision |
| :--- | :--- | :--- | :--- |
| **Architecture** | **LLaMA 3 / Mistral style** (RoPE, RMSNorm, SwiGLU) | **GPT-2 style** (Absolute Positional Embeddings, LayerNorm, GELU) | Absolute position embeddings do not generalize across varying sequence lengths and cannot capture relative token distances naturally. RMSNorm + SwiGLU is mathematically proven in modern literature to train faster with higher loss stability. |
| **Tokenizer Vocab** | **Custom 4,096 BPE** | **Standard GPT-2 (50,257) or LLaMA-3 (128,000)** | In a small model ($d=192$), a 50k vocab matrix consumes $50,257 \times 192 \approx 9.65\text{M}$ parameters just for the embedding layer! That would mean **80% of our model is a lookup table** rather than reasoning layers. A 4k vocab keeps the embedding table small (~786k params) so the parameters live inside the transformer attention blocks. |
| **Dataset Selection** | **TinyStories (Domain-Constrained)** | **OpenWebText / Common Crawl / Wikipedia** | High-entropy general web text requires billions of parameters to avoid generating gibberish. TinyStories has structured syntax and simple concepts, allowing a 2.5M model to produce coherent English stories. |
| **Attention Mechanism** | **Standard MHA + PyTorch SDPA** | **Sliding Window / FlashAttention v2 CUDA Kernels** | At sequence length 256, standard quadratic attention matrix ($256 \times 256$) is tiny (negligible memory). PyTorch's native SDPA already dispatches to optimized C++/CUDA kernels without requiring external custom C++ compiler dependencies on Windows. |
| **Parameter Size** | **~2.56M Parameters (4 layers, dim 192)** | **>50M - 100M Parameters** | Fits completely in 4GB VRAM with FP16, takes less than 1.5GB active VRAM, and completes meaningful training runs in minutes to an hour on an RTX 3050 laptop rather than days on cloud clusters. |
| **Fine-Tuning Method** | **Pre-training from scratch** | **LoRA / QLoRA fine-tuning of an existing base model** | The assignment was to build an SLM *from scratch* to understand every mathematical primitive, tensor operation, and training dynamic rather than wrapping an existing HuggingFace checkpoint. |
| **Weight Tying** | **Tied Input & Output Weights** | **Untied (Independent Head)** | Saves $4,096 \times 192 = 786,432$ parameters (~23% of the model). Weight tying acts as a regularizer in small language models, preventing the output projection from overfitting. |
| **Position Encoding** | **RoPE (Rotary Embeddings)** | **ALiBi / NoPE** | RoPE is the universal standard for modern open-weights LLMs (Llama, Gemma, Qwen, Mistral). ALiBi adds a linear bias directly to attention scores which simplifies length extrapolation, but RoPE achieves higher syntactic accuracy in dense language modelling. |

---

## 4. Key Metrics & Architecture Parameters

- **Trainable Parameters**: **2,557,632** (~2.56M)
- **Vocabulary Size**: 4,096
- **Context Length ($L$)**: 256 tokens
- **Hidden Dimension ($d_{\text{model}}$)**: 192
- **Decoder Layers**: 4
- **Attention Heads**: 6 (Head dimension = 32)
- **SwiGLU FFN Hidden Dimension**: 512
- **VRAM Footprint during Training**: ~1.2 GB (well within 4.0 GB VRAM limit)
- **Training Throughput**: ~8,000–12,000 tokens/sec on RTX 3050 Laptop GPU

---

## 5. Suggested Presentation Outline (10-15 Minutes)

1. **Slide 1: Problem Statement & Motivation**
   - Why SLMs? Why TinyStories? How small can a model be and still generate coherent English?
2. **Slide 2: Architecture Blueprint (Micro-LLaMA vs. GPT-2)**
   - Show the diagram: RMSNorm $\rightarrow$ RoPE Attention $\rightarrow$ RMSNorm $\rightarrow$ SwiGLU.
   - Explain why we chose modern LLaMA primitives over classic GPT-2.
3. **Slide 3: Tokenizer Engineering & The "Vocab Size Trap"**
   - Explain why a 50k tokenizer kills small models and why 4k is mathematically ideal.
4. **Slide 4: Systems Engineering (RTX 3050 4GB Constraints)**
   - FP16 Mixed Precision, memory-efficient dataloading, and weight tying.
5. **Slide 5: Methods Considered But Rejected**
   - Present the comparison table (LoRA vs Pretrain, GPT-2 vs LLaMA, WebText vs TinyStories).
6. **Slide 6: Live Demo / Inference Output**
   - Run `python src/generate.py` with custom prompts and show generation output.
