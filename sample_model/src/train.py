"""
High-performance PyTorch training script for Micro-LLaMA SLM on TinyStories.
Optimized for NVIDIA RTX 3050 (4GB VRAM):
- Mixed Precision (FP16 AMP)
- Gradient Clipping
- Cosine Annealing Learning Rate with linear warmup
- Live story generation check during training
- Checkpoint saving
"""
import os
import math
import time
import dataclasses
import torch
from tokenizers import Tokenizer
from src.model.llama import MicroLlamaForCausalLM, LlamaConfig
from src.data.dataset import get_dataloader

def train_slm(
    corpus_path: str = "data/tinystories_corpus.txt",
    tokenizer_path: str = "data/tinystories_tokenizer.json",
    save_dir: str = "checkpoints",
    epochs: int = 3,
    batch_size: int = 32,
    max_seq_len: int = 256,
    lr: float = 8e-4,
    min_lr: float = 8e-5,
    warmup_steps: int = 200,
    weight_decay: float = 0.05,
    eval_interval: int = 250,
    max_tokens: int = 15_000_000,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Micro-LLaMA Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Initialize Tokenizer & Model
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    config = LlamaConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dim=192,
        n_layers=4,
        n_heads=6,
        n_kv_heads=6,
        hidden_dim=512,
        dropout=0.05
    )
    
    model = MicroLlamaForCausalLM(config).to(device)
    total_params = model.get_num_params()
    print(f"Initialized Micro-LLaMA: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # 2. Dataloader
    dataloader = get_dataloader(
        data_file=corpus_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_tokens=max_tokens,
        shuffle=True
    )

    total_steps = len(dataloader) * epochs
    print(f"Total training steps: {total_steps} (Epochs: {epochs}, Batches/epoch: {len(dataloader)})")

    # 3. Optimizer & Scaler
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Learning Rate Schedule function
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr - min_lr)

    def sample_story(prompt: str = "Once upon a time, Lily found a"):
        tokens = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        out = model.generate(
            input_ids,
            max_new_tokens=40,
            temperature=0.75,
            top_k=30,
            repetition_penalty=1.15,
            eos_token_id=eos_id
        )
        return tokenizer.decode(out[0].tolist())

    # 5. Training Loop
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader, start=1):
            global_step += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Update LR
            curr_lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
                logits, loss = model(x, labels=y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Logging & periodic generation test
            if global_step % 50 == 0 or global_step == 1:
                avg_loss = running_loss / (50 if global_step % 50 == 0 else 1)
                tokens_per_sec = (batch_size * max_seq_len * 50) / max(1.0, time.time() - start_time) if global_step > 1 else 0
                print(f"[Epoch {epoch}/{epochs} | Step {global_step}/{total_steps}] Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) | LR: {curr_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")
                running_loss = 0.0
                start_time = time.time()

            if global_step % eval_interval == 0:
                print("\n--- Model Live Generation Sample ---")
                sample_output = sample_story()
                print(f"{sample_output}\n------------------------------------\n")
                
                # Checkpoint saving
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    ckpt_path = os.path.join(save_dir, "best_model.pt")
                    torch.save({
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "loss": best_loss,
                        "config": dataclasses.asdict(config)
                    }, ckpt_path)
                    print(f"Saved best model checkpoint to {ckpt_path} (Loss: {best_loss:.4f})")

    # Final save
    final_ckpt = os.path.join(save_dir, "latest_model.pt")
    torch.save({
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "loss": loss.item(),
        "config": dataclasses.asdict(config)
    }, final_ckpt)
    print(f"\nTraining completed! Saved latest checkpoint to {final_ckpt}")

if __name__ == "__main__":
    train_slm()
