"""
LIVE INTERACTIVE TOKEN TESTER
Enter any sentence to see how your trained TinyStories 4,096 BPE tokenizer chops it up into tokens!
"""
import sys
from tokenizers import Tokenizer

def live_token_explorer():
    tokenizer_path = "data/tinystories_tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print("=" * 70)
    print("      TINYSTORIES BPE TOKENIZER — LIVE TOKEN EXPLORER")
    print(f"      Loaded: {tokenizer_path} (Vocab Size: {tokenizer.get_vocab_size()})")
    print("=" * 70)

    test_sentences = [
        "Once upon a time, there was a little girl named Lily.",
        "She found a shiny yellow bell in the garden.",
        "Supercalifragilisticexpialidocious"  # An unseen long crazy word to show sub-words!
    ]

    for sentence in test_sentences:
        print(f"\n[INPUT TEXT]: '{sentence}'")
        encoded = tokenizer.encode(sentence)
        ids = encoded.ids
        tokens = encoded.tokens

        print(f"-> Total Tokens: {len(ids)}")
        print(f"-> Token IDs   : {ids}")
        print("-> Sub-word Pieces:")
        
        # Display each piece cleanly
        display_pieces = []
        for t in tokens:
            # Replace special BPE byte space representation with readable space
            clean_t = t.replace('Ġ', ' ')
            display_pieces.append(f"[{clean_t}]")
            
        print("   " + " ".join(display_pieces))
        
        decoded = tokenizer.decode(ids)
        print(f"-> Decoded Back: '{decoded}'")
        print("-" * 70)

if __name__ == "__main__":
    live_token_explorer()
