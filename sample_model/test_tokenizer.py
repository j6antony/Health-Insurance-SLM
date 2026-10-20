import os
import sys

# Ensure UTF-8 output encoding for special BPE whitespace characters in Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tokenizers import Tokenizer

def test_tokenizer(tokenizer_path: str = "data/tokenizer.json"):
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Run train_tokenizer.py first.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print("=" * 60)
    print(f"TOKENIZER LOADED (Vocabulary Size: {tokenizer.get_vocab_size()})")
    print("=" * 60)

    sample_code = """def calculate_area(radius: float) -> float:
    # Compute circular area
    pi = 3.14159
    return pi * (radius ** 2)"""

    print("\n[ORIGINAL PYTHON CODE]:")
    print(sample_code)

    # Encode
    encoded = tokenizer.encode(sample_code)
    tokens = encoded.tokens
    ids = encoded.ids

    print("\n[TOKENIZED PIECES (Sub-words)]:")
    print(tokens)

    print("\n[NUMERICAL TOKEN IDs (Fed into PyTorch)]:")
    print(ids)
    print(f"Total Tokens: {len(ids)}")

    # Decode
    decoded = tokenizer.decode(ids)
    print("\n[DECODED BACK TO STRING]:")
    print(decoded)
    print("=" * 60)

if __name__ == "__main__":
    test_tokenizer()
