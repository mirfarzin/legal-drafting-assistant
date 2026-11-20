# Project Limitations and Constraints

This document provides a critical analysis of the technical boundaries, design trade-offs, and operational limitations of the **Legal Drafting Assistant**.

---

## 1. Dataset Scale and Diversity Constraints

1. **Corpus Size:**
   - The current benchmark dataset contains 50 curated clauses (10 clauses per category). While sufficient for demonstrating algorithmic mechanics, prompt masking, and constrained decoding, real-world commercial contracts exhibit vast stylistic diversity and structural complexity that cannot be fully captured in 50 sentences.
2. **Category Scope:**
   - Only 5 standard commercial contract categories are currently recognized (`CONFIDENTIALITY`, `TERMINATION`, `GOVERNING_LAW`, `INDEMNIFICATION`, `PAYMENT`). Real commercial agreements contain dozens of specialized clauses (e.g., Force Majeure, IP Assignment, Non-Compete, Severability, Escrow).
3. **Synthetic Demonstration Data:**
   - Clauses were authored as standardized, clean English legal sentences without OCR artifacts, messy formatting, archaic legalese, or non-standard punctuation. Real-world legal documents often suffer from inconsistent typography and complex multi-page paragraph structures.

---

## 2. N-Gram Language Model Limitations

1. **Fixed Context Window:**
   - The n-gram predictor operates on an $n=3$ (trigram) horizon. It cannot maintain long-range semantic coherence across a multi-sentence clause or recall definitions established earlier in a contract.
2. **Sparsity & Out-of-Vocabulary (OOV) Terms:**
   - Although Add-$\alpha$ smoothing and backoff mitigate zero-frequency probabilities, the model cannot generalize to legal terminology absent from the initial training corpus.
3. **Syntactic vs. Semantic Generation:**
   - The n-gram generator operates purely on local lexical statistics without understanding substantive legal obligations or unintended liabilities.

---

## 3. GPT-2 Generative Classification Constraints

1. **Model Capacity & Pretraining Domain:**
   - Base GPT-2 was pretrained on generic web text (WebText), not specialized legal case law or commercial contract corpuses (like Edgar or LexisNexis). While fine-tuning effectively aligns the model to clause categorization, it relies on parameter updates rather than deep domain pretraining.
2. **Confidence Calibration:**
   - Softmax probabilities over constrained label logits can become overconfident (e.g. >95% confidence) even when an ambiguous clause spans multiple legal concepts (e.g., a clause addressing both termination notice and final severance payment).
3. **Binary Masking Boundaries:**
   - Prompt loss masking (`labels = -100`) focuses gradient updates strictly on the label token. If the input text contains contradictory clauses or adversarial text, the model may force a classification into one of the 5 allowed categories without an "Unknown / Mixed" fallback.

---

## 4. Educational Prototype Status & Legal Disclaimer

> [!CAUTION]
> **CRITICAL LEGAL NOTICE:**
> The Legal Drafting Assistant is strictly an **educational prototype** developed for the askThili *Language Models from Scratch* capstone track.
> 
> - **NOT LEGAL ADVICE:** The generated text continuations and predicted clause types are automated statistical outputs and do NOT constitute legal advice, contract review, or risk assessment.
> - **HUMAN IN THE LOOP:** No contract clause drafted or categorized by this system should ever be executed without thorough review by a qualified legal professional licensed in the relevant jurisdiction.
