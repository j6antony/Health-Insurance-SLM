# TinyStories Micro-LLaMA (SLM)

A pure PyTorch, from-scratch Small Language Model (SLM) trained on the **TinyStories** dataset (`roneneldan/TinyStories`).

---

## Architecture Highlights (~2.56M Parameters)
- **Architecture**: Modern LLaMA / Mistral Decoder-only Transformer
- **Positional Encoding**: Rotary Positional Embeddings (RoPE)
- **Normalization**: RMSNorm (Root Mean Square LayerNorm)
- **Feed-Forward**: SwiGLU Gated Activation ($SiLU(xW_1) \odot xW_2)W_3$)
- **Attention**: Scaled Dot-Product Attention with causal masking (`F.scaled_dot_product_attention`)
- **Tied Embeddings**: Output projection shares weights with input token embeddings
- **Vocabulary**: Custom Byte-Level BPE tokenizer trained specifically on TinyStories (4,096 tokens)

| Hyperparameter | Value |
| :--- | :--- |
| `vocab_size` | 4,096 |
| `max_seq_len` | 256 |
| `dim` ($d_{model}$) | 192 |
| `n_layers` | 4 |
| `n_heads` | 6 |
| `n_kv_heads` | 6 |
| `hidden_dim` | 512 |
| `params` | **2,557,632 (~2.56M)** |

---

## Project Structure

```
d:/l/fun/
├── data/
│   ├── tinystories_corpus.txt       # Streamed TinyStories dataset subset (~50,000 stories, ~44MB)
│   └── tinystories_tokenizer.json   # Trained custom 4096-vocab Byte-Level BPE tokenizer
├── docs/
│   └── specs/
│       └── tinystories_slm_spec.md  # Architectural specification & math formulation
├── src/
│   ├── model/
│   │   └── llama.py                 # Pure PyTorch Micro-LLaMA implementation (RMSNorm, RoPE, SwiGLU)
│   ├── tokenizer/
│   │   ├── train_tinystories_tokenizer.py  # Streams dataset & trains BPE tokenizer
│   │   └── test_tinystories_tokenizer.py   # Verification test for tokenizer
│   ├── data/
│   │   └── dataset.py               # Tokenizes and batches stories into causal sequences
│   ├── train.py                     # FP16 mixed precision training loop with cosine schedule
│   └── generate.py                  # Interactive CLI for temperature, top-k, and top-p sampling
└── checkpoints/
    ├── best_model.pt                # Checkpoint with lowest validation/training loss
    └── latest_model.pt              # Most recent model state
```

---

## Quickstart

### 1. Test the Tokenizer
```bash
python src/tokenizer/test_tinystories_tokenizer.py
```

### 2. Verify the Model Forward Pass & Parameter Count
```bash
python src/model/llama.py
```

### 3. Run Training
Train the model on your GPU:
```bash
python src/train.py
```
*(By default trains on ~15M tokens with FP16 on CUDA, logging loss and periodically outputting generated story samples.)*

### 4. Generate Stories (Inference CLI)
```bash
python src/generate.py --prompt "Once upon a time, Lily found a little blue bird" --temp 0.8 --top_k 40
```
