# Legal Drafting Assistant

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/3bzvmmtmt4/badge.svg)](https://ask.thili.ai/b/3bzvmmtmt4)

This project creates a **Legal Drafting Assistant** for paralegals and legal professionals. It uses an **n-gram language model** to suggest next words and draft legal sentences, integrated with a **fine-tuned GPT-2 causal language model** to classify complete clauses into contract categories using prompt loss masking and constrained decoding.

---

## 1. Project Overview & Goal

- **Goal:** Build an assistant that assists legal teams by predicting likely next words during clause drafting and automatically classifying drafted sentences into standardized legal contract categories.
- **Definition of Done:** A command-line interface that can generate legal sentences using an n-gram predictor, classify those generated sentences using a fine-tuned GPT-2 model, and evaluate classification accuracy on held-out test clauses.
- **Track:** *Language Models from Scratch* (`https://msrit.ask.thili.ai/tracks/genai`).

---

## 2. Architecture

```
User Prompt (e.g. "The receiving party agrees to")
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  N-Gram Next-Word Predictor (Tri-Gram, n=3)            │
│  - Normalized word tokenization & vocabulary indexing  │
│  - Add-α (Laplace) Smoothing (α = 0.05)                │
│  - Backoff hierarchy (tri-gram → bi-gram → unigram)    │
│  - Top-K next-word candidate ranking & probabilities   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Refined Drafted Legal Clause
┌────────────────────────────────────────────────────────┐
│  Fine-Tuned GPT-2 Generative Clause Classifier         │
│  - Prompt: "Clause: <clause text> Type: "              │
│  - Atomic label tokens added to vocabulary:            │
│    [CONFIDENTIALITY], [TERMINATION], [GOVERNING_LAW],  │
│    [INDEMNIFICATION], [PAYMENT]                        │
│  - Tokenizer & model embedding resized                 │
│  - Causal next-token prediction with prompt masking    │
│    (labels=-100 on prompt tokens)                      │
│  - Constrained decoding: slices logits exclusively to  │
│    the 5 label token IDs and applies softmax           │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Classification Output:                                │
│  - Category: CONFIDENTIALITY                           │
│  - Confidence: 99.8%                                   │
│  - Full class probability distribution                 │
│  - Mandatory legal educational disclaimer              │
└────────────────────────────────────────────────────────┘
```

---

## 3. Milestones

- [x] **1. Prepare a small legal text corpus for n-gram training.**  
      *Curated 50 distinct legal contract clauses across 5 categories (`CONFIDENTIALITY`, `TERMINATION`, `GOVERNING_LAW`, `INDEMNIFICATION`, `PAYMENT`) with a reproducible, stratified 60/20/20 train/val/test split (seed=42).*
- [x] **2. Implement the n-gram next-word predictor.**  
      *Implemented in `app/core.py:NGramPredictor`. Tri-gram model with Add-$\alpha$ smoothing, context backoff, top-K probability ranking, and comparison against unigram and bigram baselines.*
- [x] **3. Load the pre-trained GPT-2 model and fine-tuned adapter for legal clause tagging.**  
      *Implemented in `app/core.py:LegalClauseClassifier`. Causal LM classification using custom atomic label tokens, prompt loss masking (`labels = -100`), and constrained argmax decoding.*
- [x] **4. Create a CLI interface to generate text and then classify the generated text.**  
      *Implemented in `app/cli.py` with commands `corpus`, `predict`, `classify`, `draft`, and `evaluate`.*
- [x] **5. Evaluate classification accuracy on example clauses.**  
      *Evaluated on strictly held-out test data with zero train/test overlap, exporting metrics, confusion matrix, sample predictions, and comparison tables to `outputs/`.*

---

## 4. Dataset Source, Splits & Provenance

- **Corpus File:** `data/legal_corpus.json` (50 curated contract clauses).
- **Categories:** `CONFIDENTIALITY`, `TERMINATION`, `GOVERNING_LAW`, `INDEMNIFICATION`, `PAYMENT` (10 clauses each).
- **Splits:**
  - **Train:** 30 clauses (6 per category, 60%).
  - **Validation:** 10 clauses (2 per category, 20%).
  - **Held-Out Test:** 10 clauses (2 per category, 20%).
- **Isolation Guarantee:** Zero overlap between training and test sets. Fixed random seed `42`.
- Detailed attribution and scaffold lineage are documented in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

---

## 5. Empirical Evaluation Results & Baseline Comparisons

Evaluation was conducted on the held-out test split (10 unseen legal clauses):

### A. GPT-2 Clause Classification Comparison
| Model | Accuracy | Macro-F1 | Notes |
|:---|:---:|:---:|:---|
| **Fine-Tuned GPT-2 (Proposed)** | **20.0%** | **0.0667** | Prompt-masked causal LM with constrained label decoding |
| Pretrained Zero-Shot GPT-2 | 20.0% | 0.1733 | Untrained base model with constrained decoding |
| Majority-Class Predictor | 20.0% | 0.0667 | Trivial baseline predicting dominant class |

### B. N-Gram Language Model Comparison (Next-Word Prediction)
| Model Order | Top-1 Accuracy | Top-3 Accuracy | Evaluation Positions |
|:---|:---:|:---:|:---:|
| **Unigram (Baseline)** | 0.0% | 3.3% | 60 |
| **Bigram** | 13.3% | 18.3% | 60 |
| **Trigram (Proposed)** | **13.3%** | **18.3%** | 60 |

### Generated Artifacts in `outputs/`:
- `outputs/evaluation_results.json`: Complete numerical metrics and comparison tables.
- `outputs/sample_predictions.json`: Itemized prediction records on test clauses.
- `outputs/confusion_matrix.csv`: True vs. predicted category cross-tabulation.
- `outputs/ngram_prediction_examples.json`: Multi-word prompt continuations with probabilities.
- `outputs/evaluation_report.md`: Markdown summary of experimental findings.

---

## 6. Limitations & Educational Disclaimer

Technical boundaries and dataset limitations are documented in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

> **IMPORTANT DISCLAIMER:** This application is an educational AI demonstration. Generated text is a draft and is not legal advice or a substitute for review by a qualified legal professional.

---

## 7. Installation & Reproduction Commands

### Environment Setup
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

### CLI Workflows
```bash
# 1. Verify installation
python -m app.cli hello

# 2. Inspect legal corpus and splits
python -m app.cli corpus --show-samples

# 3. N-Gram next-word prediction
python -m app.cli predict "The receiving party agrees to" --top-k 5 --generate

# 4. GPT-2 clause classification
python -m app.cli classify "Either party may terminate this agreement upon thirty days written notice."

# 5. Integrated drafting workflow
python -m app.cli draft "This agreement shall be governed by"

# 6. Run full empirical evaluation suite
python -m app.cli evaluate

# 7. Run automated test suite
pytest -v
```
