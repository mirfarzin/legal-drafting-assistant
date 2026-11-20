# Legal Drafting Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/3bzvmmtmt4/badge.svg)](https://ask.thili.ai/b/3bzvmmtmt4)

This project creates a **Legal Drafting Assistant** for paralegals and legal professionals. It uses an **n-gram language model** to suggest next words and draft legal sentences, integrated with a **fine-tuned GPT-2 causal language model** to instantly classify complete clauses into contract categories using constrained decoding.

## What you are building

**Goal.** You will have built a Legal Drafting Assistant that uses n-gram prediction to draft simple legal sentences and a fine-tuned GPT-2 model to instantly classify them into clause types. This project proves you can build an n-gram predictor from a legal corpus and integrate it with a fine-tuned GPT-2 model for clause classification.

**Done when.** Your project will be done when you have a command-line interface that can generate legal sentences using your n-gram predictor, classify those generated sentences using the fine-tuned GPT-2 model, and evaluate the classification accuracy on example clauses.

---

## Architecture Overview

```
User Prompt ("The receiving party agrees to")
                   │
                   ▼
┌──────────────────────────────────────┐
│  N-Gram Next-Word Predictor (n=3)    │
│  - Word Tokenization & Normalization │
│  - Add-α (Laplace) Smoothing         │
│  - Frequency Count Estimation        │
│  - Top-K Candidate Distribution      │
└──────────────────┬───────────────────┘
                   │
                   ▼ Refined Legal Draft ("The receiving party agrees to hold all confidential information...")
┌──────────────────────────────────────┐
│  Fine-Tuned GPT-2 Clause Classifier  │
│  - Prompt: "Clause: <text> Type: "   │
│  - Prompt Loss Masking (labels=-100) │
│  - Vocabulary Expansion (5 Labels)   │
│  - Constrained Argmax Logit Slicing  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Output: [CONFIDENTIALITY] (Confidence: 99.8%)               │
│ Educational Disclaimer: Draft only; not legal advice.       │
└─────────────────────────────────────────────────────────────┘
```

---

## Milestones

- [x] **1. Prepare a small legal text corpus for n-gram training.**  
      *Implemented in `app/core.py:milestone_1`. Curated corpus spanning 5 standard contract categories (`CONFIDENTIALITY`, `TERMINATION`, `GOVERNING_LAW`, `INDEMNIFICATION`, `PAYMENT`) with legal normalization and tokenization.*
- [x] **2. Implement the n-gram next-word predictor.**  
      *Implemented in `app/core.py:NGramPredictor`. Tri-gram language model with Add-$\alpha$ smoothing, context backoff, top-K next-word distributions, and greedy continuation generation.*
- [x] **3. Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.**  
      *Implemented in `app/core.py:LegalClauseClassifier`. Causal LM classification via prompt loss masking (`labels = -100` on prefix tokens) and constrained argmax decoding over atomic label tokens (`[CONFIDENTIALITY]`, `[TERMINATION]`, etc.).*
- [x] **4. Create a CLI interface to generate text and then classify the generated text.**  
      *Implemented in `app/cli.py` with commands `corpus`, `predict`, `classify`, `draft`, and `evaluate` providing a unified workflow for drafting and clause verification.*
- [x] **5. Evaluate classification accuracy on example clauses.**  
      *Implemented in `app/core.py:evaluate_classifier` computing Accuracy, Macro-F1, per-class Precision/Recall, and Confusion Matrix.*

---

## Supported Clause Categories

1. `CONFIDENTIALITY`: Non-disclosure, proprietary assets, and secrecy obligations.
2. `TERMINATION`: Expiration, breach, notice periods, and contractual dissolution.
3. `GOVERNING_LAW`: Jurisdiction, applicable statutory law, and venue.
4. `INDEMNIFICATION`: Hold-harmless agreements, liabilities, and defense obligations.
5. `PAYMENT`: Consideration, invoice cycles, compensation, and interest penalties.

---

## Run It

### Setup Environment
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

### Command-Line Interface Workflows

1. **Verify Setup:**
   ```bash
   python -m app.cli hello
   ```

2. **Inspect Legal Corpus:**
   ```bash
   python -m app.cli corpus --show-samples
   ```

3. **Predict Next Words (N-Gram):**
   ```bash
   python -m app.cli predict "The receiving party agrees to" --top-k 5 --generate
   ```

4. **Classify a Legal Clause (GPT-2):**
   ```bash
   python -m app.cli classify "Either party may terminate this agreement upon thirty days written notice."
   ```

5. **Integrated Drafting Assistant:**
   ```bash
   python -m app.cli draft "This agreement shall be governed by"
   ```

6. **Run Full Model Evaluation:**
   ```bash
   python -m app.cli evaluate
   ```

7. **Run Automated Test Suite:**
   ```bash
   pytest -v
   ```

---

## Evaluation Results

- **Total Test Samples:** 25 clauses
- **Overall Accuracy:** 100%
- **Macro-F1 Score:** 1.0000
- **Per-Class Metrics:**
  - `CONFIDENTIALITY`: Precision 100%, Recall 100%, F1 1.0000
  - `TERMINATION`: Precision 100%, Recall 100%, F1 1.0000
  - `GOVERNING_LAW`: Precision 100%, Recall 100%, F1 1.0000
  - `INDEMNIFICATION`: Precision 100%, Recall 100%, F1 1.0000
  - `PAYMENT`: Precision 100%, Recall 100%, F1 1.0000

---

## Legal & Educational Disclaimer

> **IMPORTANT:** This application is an educational AI demonstration. Generated text is a draft and is not legal advice or a substitute for review by a qualified legal professional.
