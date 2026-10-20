"""
CLI test script for TinyStories custom BPE tokenizer.
Encodes sample story prompts and verifies token decoding and compression.
"""
import sys
from tokenizers import Tokenizer

def test_tinystories_tokenizer(tokenizer_path: str = "data/tinystories_tokenizer.json"):
    print("=" * 65)
    print("   TINYSTORIES BPE TOKENIZER TEST & INSPECTION")
    print("=" * 65)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded Tokenizer: {tokenizer_path}")
    print(f"Total Vocabulary Size: {tokenizer.get_vocab_size()} tokens\n")

    sample_story = (
        "Once upon a time, in a small green village, there was a little boy named Timmy. "
        "Timmy loved to play with his blue ball in the park. One sunny morning, he met a friendly dog."
    )

    encoded = tokenizer.encode(sample_story)
    ids = encoded.ids

    print("--- [SAMPLE INPUT STORY] ---")
    print(sample_story)
    print("\n--- [TOKEN ENCODING STATS] ---")
    print(f"Character length : {len(sample_story)} chars")
    print(f"Token count      : {len(ids)} tokens")
    print(f"Compression ratio: {len(sample_story) / len(ids):.2f} chars/token")
    print(f"First 15 Token IDs: {ids[:15]}...")

    decoded = tokenizer.decode(ids)
    print("\n--- [DECODED TEXT] ---")
    print(decoded)
    assert decoded == sample_story, "Decoded text does not match original!"
    print("\n[SUCCESS] Tokenizer round-trip match verified!")

if __name__ == "__main__":
    test_tinystories_tokenizer()
