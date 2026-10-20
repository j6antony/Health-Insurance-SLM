"""
LIVE JSON EVOLUTION DEMONSTRATOR (FOR TEAM PRESENTATION)
Shows:
1. The JSON BEFORE BPE starts (Only base alphabet, merges list is EMPTY []).
2. Live BPE merges happening step-by-step.
3. The JSON AFTER BPE finishes (New tokens added to "vocab", and each step recorded in "merges").
"""
import json
import time
from collections import Counter

def run_json_bpe_evolution_demo():
    print("=" * 75)
    print("       BPE TOKENIZER JSON EVOLUTION: BEFORE VS. AFTER DEMO       ")
    print("=" * 75)

    corpus = [
        "the cat saw the dog",
        "the dog saw the cat",
        "the cat sat on the mat"
    ]

    print("\n[OUR TRAINING DATASET]:")
    for i, line in enumerate(corpus, 1):
        print(f"  Sentence {i}: '{line}'")

    # ---------------------------------------------------------
    # STAGE 1: THE JSON *BEFORE* BPE TRAINING
    # ---------------------------------------------------------
    # Step 0: Base alphabet (characters in our corpus)
    initial_chars = sorted(list(set("".join(corpus).replace(" ", ""))))
    initial_chars.append("</w>")  # end-of-word marker

    # Build the initial JSON object
    tokenizer_json = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {char: idx for idx, char in enumerate(initial_chars)},
            "merges": []  # <--- NOTICE: EMPTY AT THE START!
        }
    }

    print("\n" + "=" * 75)
    print(">>> 1. LOOK AT THE JSON FILE *BEFORE* BPE RUNS:")
    print("=" * 75)
    print("At the start, the JSON ONLY contains single letters!")
    print("Notice that 'merges' is an EMPTY list [] because no words have merged yet:")
    print("\n" + json.dumps({
        "model_type": "BPE",
        "vocab_sample": dict(list(tokenizer_json["model"]["vocab"].items())[:6]),
        "total_vocab_count": len(tokenizer_json["model"]["vocab"]),
        "merges": tokenizer_json["model"]["merges"]
    }, indent=2))

    # ---------------------------------------------------------
    # STAGE 2: RUNNING BPE MERGES & UPDATING THE JSON
    # ---------------------------------------------------------
    word_freqs = Counter()
    for sentence in corpus:
        for word in sentence.split():
            word_freqs[tuple(list(word) + ["</w>"])] += 1

    num_merges = 5
    print("\n" + "=" * 75)
    print(f">>> 2. RUNNING BPE ALGORITHM ({num_merges} MERGES LIVE)...")
    print("=" * 75)

    for step in range(1, num_merges + 1):
        pair_counts = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq

        if not pair_counts:
            break

        best_pair, highest_count = pair_counts.most_common(1)[0]
        merged_token = "".join(best_pair)
        merge_rule_string = f"{best_pair[0]} {best_pair[1]}"

        # 1. Add to merges list in JSON
        tokenizer_json["model"]["merges"].append(merge_rule_string)
        
        # 2. Add new merged token to vocab in JSON with next ID
        next_id = len(tokenizer_json["model"]["vocab"])
        tokenizer_json["model"]["vocab"][merged_token] = next_id

        # Update word representation
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

        print(f"  [Merge #{step}] Winner pair: {best_pair} (saw {highest_count} times) -> Added '{merged_token}' (ID: {next_id}) to JSON!")

    # ---------------------------------------------------------
    # STAGE 3: THE JSON *AFTER* BPE FINISHES
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print(">>> 3. LOOK AT THE JSON FILE *AFTER* BPE FINISHES:")
    print("=" * 75)
    print("The JSON has evolved! Two critical things changed:")
    print("1. 'vocab' now includes the newly learned full words!")
    print("2. 'merges' now contains the exact recipe of merges!")
    print("\n" + json.dumps({
        "model_type": "BPE",
        "newly_added_vocab": {k: v for k, v in tokenizer_json["model"]["vocab"].items() if v >= len(initial_chars)},
        "total_final_vocab": len(tokenizer_json["model"]["vocab"]),
        "merges_recipe": tokenizer_json["model"]["merges"]
    }, indent=2))

    print("\n" + "-" * 75)
    print(">>> 4. WHAT TO SHOW & EXPLAIN TO YOUR TEAM:")
    print("-" * 75)
    print("1. Open data/tinystories_tokenizer.json in your editor to show them the real file.")
    print("2. Scroll to 'model' -> 'vocab': show them how letters have low IDs, while words like 'Once' have higher IDs.")
    print("3. Scroll to 'model' -> 'merges': show them the exact merge list that BPE wrote!")
    print("-" * 75)

if __name__ == "__main__":
    run_json_bpe_evolution_demo()
