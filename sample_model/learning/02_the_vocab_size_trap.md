# Module 2: The "Vocab Size Trap" in Small Language Models (SLMs)

---

## 1. What is an Embedding Table?

Before a token ID (like `439` for `"Once"`) enters the Transformer neural network, it must be turned into a list of numbers called a **Vector (or Embedding)**.

The model stores an **Embedding Matrix ($W_{\text{embed}}$)**:
$$\text{Dimensions} = (\text{Vocabulary Size}) \times (\text{Hidden Dimension } d)$$

In our model:
- $\text{Hidden Dimension } d = 192$

Every token gets its own row of 192 numbers.

---

## 2. The Math: 50,000 Vocab vs. 4,096 Vocab

Let's calculate the parameter cost of both options:

### Option A: Standard GPT-2 Vocab (50,257 tokens)
$$\text{Parameters} = 50,257 \times 192 = \mathbf{9,649,344 \text{ parameters (~9.65 Million)}}$$

If our total model budget is ~12 Million parameters:
- **Embedding Table**: 9.65M parameters (**80% of the entire model!**)
- **Transformer Brain (Layers, Attention, MLP)**: Only 2.35M parameters (**20%**).
- **The Result**: The model wastes 80% of its memory just storing a giant dictionary lookup table! It has almost no parameters left to actually reason, remember characters, or construct coherent story plots!

---

### Option B: Our Choice: 4,096 Vocab
$$\text{Parameters} = 4,096 \times 192 = \mathbf{786,432 \text{ parameters (~0.78 Million)}}$$

Now look at the distribution:
- **Embedding Table**: 0.78M parameters (**~30% of model**).
- **Transformer Brain (4 Layers, Attention, SwiGLU)**: 1.77M parameters (**~70% of model**).
- **The Result**: The vast majority of our model's capacity is spent on **learning grammar, logic, and storytelling**, not storing lookup words!

---

## 3. Why Weight Tying Solves Another 786,432 Parameters

At the very end of the network, the model must take its internal state (192 numbers) and project it back into 4,096 probabilities (one for each word in the dictionary).

That requires an **Output Head Matrix ($W_{\text{out}}$)** of size $192 \times 4096$.

* **Normal way**: Create a brand new matrix $\rightarrow$ costs another 786,432 parameters.
* **What WE Did (Weight Tying)**: We set:
  $$W_{\text{out}} = W_{\text{embed}}^T$$
  We reuse the exact same weights!
* **Benefits**:
  1. Saves ~786k parameters (~23% of total model size).
  2. Prevents the model from overfitting.

---

## 4. Presentation Takeaway for Your Team

> *"We deliberately chose a 4,096 vocabulary instead of the standard 50,000 GPT-2 vocabulary. In a small language model with hidden dimension 192, a 50k vocabulary creates a 9.6M parameter embedding table, consuming 80% of the parameter budget on a static dictionary. By using 4k tokens and weight tying, over 70% of our parameters remain dedicated to the Transformer attention and feed-forward reasoning layers."*
