# Project Provenance and Attribution

This document details the origin, lineage, dependencies, and author contributions of the **Legal Drafting Assistant** repository.

---

## 1. Inherited Scaffold vs. Original Implementation

### A. Inherited from askThili Starter Scaffold
The repository structure was initialized from the official `scaffold.zip` provided by askThili for the **Legal Drafting Assistant** project under the *Language Models from Scratch* track (`https://msrit.ask.thili.ai/tracks/genai`):

- **Project Metadata & Scaffolding:**
  - `pytest.ini`: Pytest configuration setting root test directory.
  - `.github/workflows/ci.yml`: GitHub Actions automated CI workflow running on `ubuntu-latest`.
  - `.gitignore`: Standard Python and local data exclusions.
  - `.env.example`: Template for optional external LLM endpoints.
  - `app/__init__.py`: Package initialization.
  - `app/llm.py`: Optional chat completion wrapper for local Ollama or cloud providers.
  - `app/storage.py`: Basic JSON key-value store.
  - `tests/test_smoke.py`: Initial sanity tests for storage, environment variables, and CLI wiring.

### B. Custom & Substantially Modified Components
All domain logic, algorithmic implementations, and evaluation pipelines in `app/core.py`, `app/cli.py`, and `tests/test_core.py` were independently implemented:

1. **N-Gram Language Model (`app/core.py:NGramPredictor`):**
   - Built from first principles following the curriculum of Drill 1 (*Build a Next-Word Predictor from Scratch*).
   - Word-level tokenization, vocabulary indexing, and special token boundary handling (`<bos>`, `<eos>`).
   - Add-$\alpha$ (Laplace) probability smoothing.
   - Dynamic context backoff (trigram $\to$ bigram $\to$ unigram).
   - Autoregressive greedy text generation and top-K candidate probability ranking.
2. **GPT-2 Generative Clause Classifier (`app/core.py:LegalClauseClassifier`):**
   - Built on concepts established in Drill 4 (*Fine-Tune GPT-2 to Tag Legal Contract Clauses*).
   - Dynamic vocabulary expansion registering 5 atomic label tokens: `[CONFIDENTIALITY]`, `[TERMINATION]`, `[GOVERNING_LAW]`, `[INDEMNIFICATION]`, and `[PAYMENT]`.
   - Tokenizer and model embedding resizing (`model.resize_token_embeddings`).
   - Sequence formatting: `"Clause: <clause text> Type: <label_token>"`.
   - Prompt loss masking setting `labels = -100` on prompt prefix tokens so gradients backpropagate strictly through the classification decision.
   - Constrained argmax decoding: logit slicing restricted exclusively to valid label token IDs to eliminate out-of-vocabulary hallucinations.
3. **Comparative Baseline Suite (`app/core.py`):**
   - Implemented `MajorityClassBaseline` and `PretrainedZeroShotBaseline` for classification.
   - Implemented unigram and bigram baselines for next-word prediction.
4. **Interactive CLI (`app/cli.py`):**
   - Implemented subcommands: `corpus`, `predict`, `classify`, `draft`, and `evaluate`.
5. **Artifact & Evaluation Pipeline (`app/core.py:run_full_evaluation_pipeline`):**
   - Automated export of empirical results, confusion matrices, and markdown evaluation reports into `outputs/`.

---

## 2. Dataset Provenance & Corpus Lineage

- **Dataset Nature:** Curated synthetic legal contract clause corpus (`data/legal_corpus.json`).
- **Composition:** 50 distinct contractual clauses (10 clauses per category) spanning:
  1. `CONFIDENTIALITY`: Non-disclosure, trade secrets, proprietary information protection.
  2. `TERMINATION`: Written notice, material breach, automatic expiration.
  3. `GOVERNING_LAW`: Jurisdiction, venue, and choice of law.
  4. `INDEMNIFICATION`: Hold-harmless, liability limitation, and third-party defense.
  5. `PAYMENT`: Compensation terms, monthly invoicing, and late interest.
- **Data Splitting & Isolation:**
  - Split: 60% Train (30 clauses), 20% Validation (10 clauses), 20% Held-Out Test (10 clauses).
  - Splitting method: Stratified by category using fixed random seed `42`.
  - Guarantee: Zero overlap between training examples and evaluation test examples.

---

## 3. External Libraries & Frameworks

| Package | Version | Purpose | License |
|:---|:---:|:---|:---|
| `torch` | 2.15.0+cpu | Core tensor computations and neural network backpropagation | Apache 2.0 / BSD |
| `transformers` | 5.23.0 | Hugging Face GPT-2 model architecture and tokenizer | Apache 2.0 |
| `datasets` | 5.2.0 | Dataset management tools | Apache 2.0 |
| `accelerate` | 1.16.0 | Hardware abstraction and device management | Apache 2.0 |
| `httpx` | 0.28.1 | HTTP transport client | BSD-3-Clause |
| `pytest` | 9.2.0 | Automated testing framework | MIT |

---

## 4. Academic & Curriculum References
- **Drill 1:** *Build a Next-Word Predictor from Scratch* (askThili, 2026) — N-gram modeling, maximum likelihood estimation, Laplace smoothing, cross-entropy.
- **Drill 4:** *Fine-Tune GPT-2 to Tag Legal Contract Clauses* (askThili, 2026) — Generative classification as next-token prediction, prompt masking, vocabulary expansion.
- **Radford et al. (2019):** *Language Models are Unsupervised Multitask Learners* (GPT-2 paper).
