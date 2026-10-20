# Module 1: Tokenization & Byte-Pair Encoding (BPE) from Absolute Zero

---

## 1. Why Do We Need Tokenization?

Computers, GPUs, and Neural Networks **cannot understand English, letters, or words**.
They only perform matrix multiplication on **numbers (tensors)**.

If a human types:
> `"Lily has a cat."`

We must convert each piece of that text into numbers:
> `[142, 38, 12, 804]`

This translation step is called **Tokenization**.

---

## 2. The 3 Ways to Convert Text to Numbers

| Method | How it works | The Fatal Flaw |
| :--- | :--- | :--- |
| **1. Word-Level** | Every distinct word gets 1 number (`cat` = 1, `dog` = 2, `Lily` = 3). | **Out of Vocabulary (`<unk>`)**: The English language has 500,000+ words. If someone types a typo (`"caat"`), slang, or new name, the model breaks and outputs `<unk>`. Also, a 500k dictionary consumes gigantic memory. |
| **2. Character-Level** | Every letter gets 1 number (`a` = 1, `b` = 2 ... `z` = 26). | **Sequences become way too long**: A 200-word story turns into 1,200 individual characters. The model wastes its parameters learning how to spell words before it can even understand the plot. |
| **3. Sub-word (BPE)** | **The Golden Middle Ground!** Common words are 1 token (`the`, `cat`). Rare or unknown words get split into syllables (`un-happy`, `z-y-lo-phone`). | **Zero unknown words!** Retains compact sequence lengths while never failing on unexpected words. |

---

## 3. How Byte-Pair Encoding (BPE) Works Step-by-Step

### The Algorithm:
1. **Start with raw characters/bytes** (our base alphabet: 256 bytes).
2. Look across the entire text corpus and **count which two adjacent characters appear together most frequently**.
3. **Merge that winning pair** into a brand-new token and add it to the vocabulary.
4. **Repeat** this merge loop until the vocabulary hits your target size (e.g., 4,096 tokens).

### Visual Step-by-Step Example:
Suppose our dataset has:
- `t h e` (seen 100 times)
- `c a t` (seen 50 times)

```
Step 0: Alphabet = ['t', 'h', 'e', 'c', 'a']
Words:
  't'  'h'  'e'
  'c'  'a'  't'

Merge 1: Most frequent pair is ('t', 'h') [100 times]
  New Token Created: 'th'
  Words:
    'th'  'e'
    'c'   'a'  't'

Merge 2: Most frequent pair is ('th', 'e') [100 times]
  New Token Created: 'the'
  Words:
    'the'
    'c'   'a'  't'
```
Now, **"the"** is a single token!

---

## 4. What We Did in Our Code

In `src/tokenizer/train_tinystories_tokenizer.py`:
- We streamed 50,000 stories from the `roneneldan/TinyStories` dataset.
- We started with **ByteLevel(add_prefix_space=False)** so every ASCII and Unicode byte is supported.
- We told the trainer: **stop when you hit 4,096 tokens**.
- Saved the final vocabulary to `data/tinystories_tokenizer.json`.

---

## 5. What We Did NOT Do & Why

- **Did NOT use standard GPT-2 tokenizer (50,257 tokens)**:
  - In a small model, a 50,000-word dictionary requires millions of parameters just for the embedding layer before the model even has a single Transformer layer!
- **Did NOT use character-level tokenization**:
  - Context window of 256 would only fit ~50 words instead of a full story.
