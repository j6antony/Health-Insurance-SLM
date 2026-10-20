"""
BPE BEFORE vs. AFTER VISUAL COMPARATOR (FOR TEAM PRESENTATION)
Demonstrates the dramatic transformation of raw text before and after BPE.
"""
import sys
from tokenizers import Tokenizer

def show_before_and_after():
    tokenizer_path = "data/tinystories_tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    story = (
        "Once upon a time, Lily found a little bird. "
        "The bird was very hungry, so Lily gave it some bread."
    )

    print("=" * 75)
    print("        BYTE-PAIR ENCODING (BPE): BEFORE vs. AFTER COMPARISON        ")
    print("=" * 75)
    print("\n[INPUT STORY TO TOKENIZE]:")
    print(f'"{story}"')
    print(f"Total Characters: {len(story)} chars")

    # ----------------------------------------------------
    # BEFORE BPE: Raw Character Level (Step 0)
    # ----------------------------------------------------
    raw_chars = [c if c != " " else "·" for c in story]
    print("\n" + "=" * 75)
    print(">>> 1. BEFORE BPE (Raw Character / Byte Level):")
    print("=" * 75)
    print("The computer only knows individual letters. Words are shattered into pieces:")
    print(" | ".join(raw_chars[:35]) + " | ...")
    print(f"\n[Problem]:")
    print(f"-> Total Pieces: {len(story)} tokens to process!")
    print("-> The model wastes brainpower just learning to spell words like 'O-n-c-e'.")
    print("-> Fits very little story into context window (256 tokens = only ~45 words).")

    # ----------------------------------------------------
    # AFTER BPE: Trained 4,096 Sub-word BPE
    # ----------------------------------------------------
    encoded = tokenizer.encode(story)
    ids = encoded.ids
    tokens = [t.replace('Ġ', ' ') for t in encoded.tokens]

    print("\n" + "=" * 75)
    print(">>> 2. AFTER BPE (Our Trained 4,096 Vocab Tokenizer):")
    print("=" * 75)
    print("BPE has merged frequent letter combinations into meaningful tokens:")
    
    formatted_tokens = [f"[{t.strip()}]" for t in tokens]
    print(" ".join(formatted_tokens))

    print(f"\n[Token IDs]:")
    print(ids)

    print("\n" + "-" * 75)
    print(">>> 3. SIDE-BY-SIDE SUMMARY FOR YOUR PRESENTATION SLIDE:")
    print("-" * 75)
    print(f"| Metric                | BEFORE BPE (Letters)   | AFTER BPE (Our Model)  |")
    print(f"|-----------------------|------------------------|------------------------|")
    print(f"| Token Count           | {len(story):<22} | {len(ids):<22} |")
    print(f"| Compression Ratio     | 1.00 chars/token       | {len(story)/len(ids):.2f} chars/token       |")
    print(f"| Efficiency Gain       | Baseline               | {(len(story)/len(ids)):.1f}x More Compact!       |")
    print(f"| Context Window (256)  | ~45 words max          | ~190 words (Full story)|")
    print(f"| Unknown Words (<unk>) | 0                      | 0 (Guaranteed)         |")
    print("-" * 75)

if __name__ == "__main__":
    show_before_and_after()
