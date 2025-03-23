Outside a long-abandoned movie theater in Maplewood sits a dusty civic sign that still reads:

> **WE ARE CLOSED UNTIL FURTHER NOTICE STAY HEALTHY**

It’s a pandemic-era holdover, frozen in time.

Rather than simply replace it, I set out to **transform it** — using anagrammatic recomposition to breathe fresh civic life into the original message.

The constraint? **Every single letter must come from the original phrase**. No additions, no deletions. Just reshuffled reality.

And the goal? To surface **grammatically correct**, **emotionally resonant**, **public-facing phrases** worthy of the sign.

---

## The Challenge

At first glance, this looks like a simple word game. In reality, it’s a brutal **neuro-symbolic constraint problem**:

- The original phrase contains **49 alphabetic characters**
- That’s a **multiset** of characters, each with a fixed frequency (e.g. 3 E’s, 2 R’s, etc.)
- We're searching for new phrases that are **exact multiset anagrams**, not loose approximations

This immediately blows up into a combinatorial space with **billions of possible word combinations**—most of which are nonsense.

To make matters harder, **language models are not good at exact letter tracking**, and symbolic anagram generators don’t produce meaningful language.

We needed a hybrid approach.

---

## The Method

The system evolved into a multi-stage pipeline:

1. **Input Normalization**  
   Normalize the original phrase by stripping out spaces and punctuation to get a 49-character multiset.

2. **Dictionary Filtering**  
   Use a large dictionary to find all words constructible from that letter pool. That gave us ~12,000 candidate English words.

3. **Seed Word Selection via Embeddings**  
   Use a large language model (LLM) to find seed words that are both meaningful and relevant to the original theme (e.g. "health," "closed," "safe").

4. **LLM-Guided Completions**  
   Feed these seeds back into the LLM with precise remaining-letter constraints and ask it to produce full-sentence completions.

5. **Fitness Evaluation**  
   For each candidate, calculate:
   - **Symbolic fitness**: how closely it matches the exact letter pool
   - **Fluency**: grammar and readability
   - **Emotional/civic tone**
   - **Combined score**: a weighted combination

6. **Semantic Reranking**  
   LLMs were then used again to **rank** the most meaningful and civic-appropriate candidates.

---

## The Math of Impossibility

Let’s talk probability. There are roughly:

- ~12,000 words you can construct from the 49-letter pool
- Assuming we try to form **5- to 8-word phrases**, the number of possible ordered combinations is astronomical
- But only a **tiny fraction** of those combinations:
  - Use all 49 letters exactly once
  - Are grammatically valid
  - Convey meaning

We estimate the probability of randomly stumbling onto a valid, meaningful anagram is less than:

> **1 in 10 million**

That’s why a brute-force approach is hopeless. And why language guidance — however imprecise — became essential.

---

## The Results

After hours of iteration and some offline processing, we found a few early promising candidates, including:

> **"Codes: Healthy future - We are in this, really"**

and

> **"Introductions refuters: we'll teach ya healthy!"**

Are they perfect? No.  
But they show that **meaning is within reach**.

---

## What’s Next

- We're now refining our pipeline with:
  - Better filtering
  - Claude and GPT dual support
  - Improved fitness scoring and retry logic
- We're also experimenting with:
  - Constrained beam search
  - Token-level control (e.g. via fine-tuning)
  - Visual mockups for real-world deployment

And yes — we **will** find an exact, meaningful anagram that works on the sign.

---

## Why This Matters

This is more than a puzzle.

It’s about using AI to find **poetry within constraints**, to explore how machines can **co-author public meaning**, and to prove that even something as rigid as a civic sign can spark creative emergence.

Constraints don’t kill creativity. They **give it form**.

Let’s keep searching.

— *David Palaitis*

