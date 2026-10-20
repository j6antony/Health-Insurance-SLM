"""
Custom Byte-Pair Encoding (BPE) Tokenizer for Micro Python SLM.
Trained directly on Python source code, preserving indentation, operators, and keywords.
"""
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_bpe_tokenizer(
    corpus_path: str = "data/python_corpus.txt",
    save_path: str = "data/tokenizer.json",
    vocab_size: int = 2048
):
    print(f"Training BPE Tokenizer on {corpus_path} (target vocab size: {vocab_size})...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Initialize empty BPE model with unknown token placeholder
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # 2. ByteLevel Pre-tokenizer: Treats raw bytes as characters, perfectly preserving 
    #    whitespace, 4-space indentations, newlines, and symbols.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 3. Special tokens configuration and full 256-byte baseline alphabet (zero unknown characters)
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    # 4. Train tokenizer on the dataset files
    tokenizer.train(files=[corpus_path], trainer=trainer)

    # 5. Save tokenizer artifact
    tokenizer.save(save_path)
    print(f"Tokenizer successfully trained and saved to {save_path} (Final Vocab Size: {tokenizer.get_vocab_size()})")

    return tokenizer

if __name__ == "__main__":
    train_bpe_tokenizer()
