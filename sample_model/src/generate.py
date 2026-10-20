"""
Interactive text generation and inference script for Micro-LLaMA SLM.
Supports greedy decoding, temperature sampling, top-k, and top-p (nucleus) filtering.
"""
import os
import argparse
import torch
from tokenizers import Tokenizer
from src.model.llama import MicroLlamaForCausalLM, LlamaConfig

def load_model_and_tokenizer(
    checkpoint_path: str = "checkpoints/best_model.pt",
    tokenizer_path: str = "data/tinystories_tokenizer.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    if not os.path.exists(checkpoint_path):
        alt_path = "checkpoints/latest_model.pt"
        if os.path.exists(alt_path):
            print(f"'{checkpoint_path}' not found, falling back to '{alt_path}'")
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Please train the model first.")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config_dict = checkpoint.get("config", {})
    config = LlamaConfig(**config_dict) if config_dict else LlamaConfig()
    
    model = MicroLlamaForCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device

def generate_story(
    model: MicroLlamaForCausalLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.75,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cuda"
) -> str:
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Identify end-of-text token ID if present
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id
        )
    
    generated_text = tokenizer.decode(out_ids[0].tolist())
    return generated_text

def interactive_cli():
    parser = argparse.ArgumentParser(description="Micro-LLaMA Story Generator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tinystories_tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little dog named Spot.", help="Prompt to start")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.75, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling")
    parser.add_argument("--rep_penalty", type=float, default=1.15, help="Repetition penalty (1.0 = no penalty)")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.tokenizer)
    print(f"\n================ MICRO-LLAMA STORY GENERATOR ================")
    print(f"Device: {device} | Temperature: {args.temp} | Top-K: {args.top_k} | Top-P: {args.top_p} | Rep-Penalty: {args.rep_penalty}")
    print(f"=============================================================\n")

    result = generate_story(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        device=device
    )
    print(f"PROMPT:\n{args.prompt}\n")
    print(f"GENERATION:\n{result}\n")

if __name__ == "__main__":
    interactive_cli()
