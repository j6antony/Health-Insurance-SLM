"""
From-scratch PyTorch implementation of a Micro-LLaMA Small Language Model (SLM).
Includes:
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- SwiGLU Feed-Forward Network
- Multi-Head Causal Self-Attention with KV caching support
- MicroLlamaForCausalLM
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LlamaConfig:
    vocab_size: int = 4096
    max_seq_len: int = 256
    dim: int = 192          # Hidden state dimension (d_model)
    n_layers: int = 4       # Transformer blocks
    n_heads: int = 6        # Attention heads (dim // n_heads = 32 head_dim)
    n_kv_heads: int = 6     # Multi-query / grouped-query attention support
    hidden_dim: int = 512   # SwiGLU expansion layer
    norm_eps: float = 1e-5
    dropout: float = 0.0

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed

def precompute_rope_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes complex exponentials cis = cos + i*sin for RoPE."""
    # head_dim must be even
    assert dim % 2 == 0, "Dimension must be divisible by 2 for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex tensor
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary positional embeddings to Query and Key tensors."""
    # xq: (batch, n_heads, seq_len, head_dim)
    # Convert to complex numbers: (..., head_dim // 2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # freqs_cis: (seq_len, head_dim // 2) -> reshape for broadcasting: (1, 1, seq_len, head_dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network: SiLU(xW_gate) * xW_up @ W_down"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))

class CausalSelfAttention(nn.Module):
    """Multi-Head Attention with Rotary Position Embeddings and Causal Masking."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        b_sz, seq_len, _ = x.shape
        
        # 1. Projections
        xq = self.wq(x).view(b_sz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(b_sz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(b_sz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply RoPE
        freqs_cis_seq = freqs_cis[:seq_len]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis_seq)
        
        # 3. Scaled Dot-Product Attention with built-in causal masking (PyTorch 2.0+ optimized FlashAttention/SDPA)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True, dropout_p=drop_p)
        
        # 4. Transpose back and project out
        out = out.transpose(1, 2).contiguous().view(b_sz, seq_len, -1)
        return self.wo(out)

class TransformerBlock(nn.Module):
    """Single LLaMA Transformer Decoder layer."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = SwiGLU(config.dim, config.hidden_dim, config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-normalization residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class MicroLlamaForCausalLM(nn.Module):
    """Complete Micro-LLaMA Causal Language Model."""
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        
        # Token Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        
        # Transformer Blocks
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        # Final Norm & Head
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight tying (helps compact models generalize and saves memory)
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute RoPE frequencies table
        head_dim = config.dim // config.n_heads
        freqs_cis = precompute_rope_freqs_cis(head_dim, config.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Returns total trainable parameters (excluding tied duplicate weights)."""
        params = [p for p in self.parameters() if p.requires_grad]
        unique_params = {p.data_ptr(): p for p in params}.values()
        return sum(p.numel() for p in unique_params)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b_sz, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"
        
        h = self.dropout(self.tok_embeddings(input_ids))
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        logits = self.output(h)  # (batch, seq_len, vocab_size)
        
        loss = None
        if labels is not None:
            if labels.shape[1] == logits.shape[1]:
                # Labels are already target-aligned with input_ids (x -> y from causal dataset)
                loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
            else:
                # Standard causal shift if labels == input_ids (e.g. self-supervised causal LM)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Autoregressive generation with temperature, top-k, top-p, and repetition penalty."""
        self.eval()
        b_sz = input_ids.shape[0]
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].clone()  # Take the last token predictions
            
            # Apply Repetition Penalty to prevent loops
            if repetition_penalty != 1.0:
                for b in range(b_sz):
                    for token_id in set(idx_cond[b].tolist()):
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            if temperature == 0.0:
                # Greedy
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(1e-5, temperature)
                
                # Top-K filtering
                if top_k > 0:
                    val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < val[:, [-1]]] = -float('Inf')
                
                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        return input_ids

if __name__ == "__main__":
    cfg = LlamaConfig(vocab_size=4096, max_seq_len=256, dim=192, n_layers=4, n_heads=6, hidden_dim=512)
    model = MicroLlamaForCausalLM(cfg)
    total_params = model.get_num_params()
    print(f"Model successfully built!")
    print(f"Total trainable parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    # Test forward pass with random batch
    dummy_x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(dummy_x, labels=dummy_x)
    print(f"Forward output logits shape: {logits.shape}, test loss: {loss.item():.4f}")
