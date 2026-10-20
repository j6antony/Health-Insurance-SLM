"""
Download a subset of TinyStories dataset and train a custom Byte-Level BPE Tokenizer.
Target vocab size: 4,096 tokens (compact, specialized for children's stories).
"""
import os
import itertools
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def prepare_corpus_and_train_tokenizer(
    num_samples: int = 50_000,
    corpus_text_path: str = "data/tinystories_corpus.txt",
    save_tokenizer_path: str = "data/tinystories_tokenizer.json",
    vocab_size: int = 4096
):
    os.makedirs(os.path.dirname(corpus_text_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_tokenizer_path), exist_ok=True)

    # 1. Download or extract sample stories from TinyStories
    if not os.path.exists(corpus_text_path) or os.path.getsize(corpus_text_path) == 0:
        print(f"Streaming {num_samples} stories from 'roneneldan/TinyStories' to build tokenizer corpus...")
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        count = 0
        with open(corpus_text_path, "w", encoding="utf-8") as f:
            for item in itertools.islice(dataset, num_samples):
                text = item.get("text", "").strip()
                if text:
                    f.write(text + "\n<|endoftext|>\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"Saved {count}/{num_samples} stories...")
        print(f"Corpus saved to {corpus_text_path} ({count} stories).")
    else:
        print(f"Using existing corpus at {corpus_text_path}.")

    # 2. Setup Byte-Level BPE Tokenizer
    print(f"Training BPE Tokenizer (target vocab size: {vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = ["<unk>", "<|endoftext|>", "<s>", "</s>", "<pad>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    # 3. Train on corpus
    tokenizer.train(files=[corpus_text_path], trainer=trainer)
    tokenizer.save(save_tokenizer_path)

    print(f"Tokenizer trained & saved to {save_tokenizer_path}")
    print(f"Actual Vocab Size: {tokenizer.get_vocab_size()}")
    
    # Quick sanity test
    test_sample = "Once upon a time, there was a little girl named Lily. She had a yellow ball."
    encoded = tokenizer.encode(test_sample)
    print("\n--- Sanity Check ---")
    print(f"Sample Text: {test_sample}")
    print(f"Number of tokens: {len(encoded.ids)}")
    print(f"Token IDs: {encoded.ids[:12]}...")
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded: {decoded}")

    return tokenizer

if __name__ == "__main__":
    prepare_corpus_and_train_tokenizer()
