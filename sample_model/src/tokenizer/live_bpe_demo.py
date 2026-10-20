"""
LIVE ACTION BPE DEMONSTRATOR
Watch Byte-Pair Encoding learn step-by-step from scratch on a small text corpus!
"""
import time
from collections import Counter

def run_live_bpe_demo():
    print("=" * 70)
    print("        WELCOME TO LIVE BPE (BYTE-PAIR ENCODING) IN ACTION!       ")
    print("=" * 70)
    
    # 1. Our tiny sample story dataset
    corpus = [
        "the cat saw the dog",
        "the dog saw the cat",
        "the cat sat on the mat"
    ]
    
    print("\n[STEP 1] OUR RAW DATASET:")
    for i, line in enumerate(corpus, 1):
        print(f"  Sentence {i}: '{line}'")
        
    # 2. Break down each word into individual character symbols (with a special end-of-word marker </w>)
    # This represents our initial vocabulary (base alphabet)
    word_freqs = Counter()
    for sentence in corpus:
        for word in sentence.split():
            # Add end-of-word marker so "cat" doesn't merge with "saw"
            tokenized_word = tuple(list(word) + ["</w>"])
            word_freqs[tokenized_word] += 1
            
    # Initial vocab
    vocab = set()
    for word in word_freqs:
        for char in word:
            vocab.add(char)
            
    print("\n" + "-" * 70)
    print(f"[STEP 2] INITIAL STATE (Starting with individual letters):")
    print(f"  Starting Vocab Size : {len(vocab)} characters")
    print(f"  Starting Characters : {sorted(list(vocab))}")
    print("\n  Words represented as split characters with frequencies:")
    for w, freq in word_freqs.items():
        print(f"    {' '.join(w)}  --> appears {freq} times")

    # Target number of merges
    num_merges = 6
    print("\n" + "=" * 70)
    print(f"[STEP 3] NOW LET'S DO {num_merges} MERGES LIVE!")
    print("=" * 70)

    for step in range(1, num_merges + 1):
        # Count all adjacent pairs in current words
        pair_counts = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq

        if not pair_counts:
            break

        # Find the single most frequent pair
        best_pair, highest_count = pair_counts.most_common(1)[0]
        merged_token = "".join(best_pair)
        
        print(f"\n--- [MERGE #{step}] ---")
        print(f"Scanning all pairs across words...")
        top_3 = pair_counts.most_common(3)
        print(f"Top frequent pairs found: {', '.join([f'{k}: {v} times' for k, v in top_3])}")
        print(f">>> WINNER: Pair {best_pair} appeared {highest_count} times!")
        print(f">>> ACTION: Merge {best_pair}  ===>  '{merged_token}'")
        
        # Add to vocab
        vocab.add(merged_token)
        print(f"New Vocab Size: {len(vocab)} tokens")

        # Update words by combining this pair
        new_word_freqs = Counter()
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
            
        word_freqs = new_word_freqs

        print("\nCurrent word representations in our dataset:")
        for w, freq in word_freqs.items():
            print(f"    {' | '.join(w)}")
            
    print("\n" + "=" * 70)
    print("[FINAL RESULT] LOOK WHAT HAPPENED:")
    print("=" * 70)
    print("Notice how frequent words like 'the</w>' became ONE SINGLE TOKEN!")
    print("While rare words still remain safely broken down into letters or syllables.")
    print("This is exactly how BPE created 'tinystories_tokenizer.json'!\n")

if __name__ == "__main__":
    run_live_bpe_demo()
