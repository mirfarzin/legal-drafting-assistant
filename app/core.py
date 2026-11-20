"""Core implementation of Legal Drafting Assistant.

Integrates:
1. An n-gram next-word predictor trained on legal contract corpora.
2. A fine-tuned GPT-2 generative classifier using prompt loss masking
   and constrained vocabulary decoding over atomic clause label tokens.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

# Standard legal clause categories and atomic label tokens
LEGAL_CATEGORIES: list[str] = [
    "CONFIDENTIALITY",
    "TERMINATION",
    "GOVERNING_LAW",
    "INDEMNIFICATION",
    "PAYMENT",
]

LABEL_TOKENS: list[str] = [f"[{cat}]" for cat in LEGAL_CATEGORIES]
LABEL_TO_CAT: dict[str, str] = {tok: cat for tok, cat in zip(LABEL_TOKENS, LEGAL_CATEGORIES)}
CAT_TO_LABEL: dict[str, str] = {cat: tok for tok, cat in zip(LEGAL_CATEGORIES, LABEL_TOKENS)}

LEGAL_DISCLAIMER: str = (
    "This application is an educational AI demonstration. Generated text is a draft "
    "and is not legal advice or a substitute for review by a qualified legal professional."
)

# Benchmark legal corpus with realistic clauses across standard contract categories
DEFAULT_LEGAL_CORPUS: list[dict[str, str]] = [
    # Confidentiality clauses
    {
        "text": "The receiving party agrees to hold all confidential information in strict confidence.",
        "category": "CONFIDENTIALITY",
    },
    {
        "text": "The receiving party shall not disclose any proprietary information to any third party without prior written consent.",
        "category": "CONFIDENTIALITY",
    },
    {
        "text": "Confidential information shall remain the exclusive property of the disclosing party at all times.",
        "category": "CONFIDENTIALITY",
    },
    {
        "text": "Each party shall protect confidential documents using the same degree of care as for its own proprietary records.",
        "category": "CONFIDENTIALITY",
    },
    {
        "text": "Upon written request the recipient shall promptly return or destroy all copies of confidential materials.",
        "category": "CONFIDENTIALITY",
    },
    # Termination clauses
    {
        "text": "Either party may terminate this agreement upon thirty days prior written notice to the other party.",
        "category": "TERMINATION",
    },
    {
        "text": "In the event of a material breach this agreement may be terminated immediately by written notice.",
        "category": "TERMINATION",
    },
    {
        "text": "This agreement shall automatically expire on the second anniversary of the effective date.",
        "category": "TERMINATION",
    },
    {
        "text": "Upon termination of this agreement all rights and obligations of the parties shall cease immediately.",
        "category": "TERMINATION",
    },
    {
        "text": "The company reserves the right to terminate employment immediately for gross misconduct.",
        "category": "TERMINATION",
    },
    # Governing Law clauses
    {
        "text": "This agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
        "category": "GOVERNING_LAW",
    },
    {
        "text": "The parties irrevocably submit to the exclusive jurisdiction of the state and federal courts located in New York.",
        "category": "GOVERNING_LAW",
    },
    {
        "text": "Any dispute arising under this contract shall be resolved exclusively through binding arbitration.",
        "category": "GOVERNING_LAW",
    },
    {
        "text": "This contract shall be interpreted under the laws of England and Wales without regard to conflict of law principles.",
        "category": "GOVERNING_LAW",
    },
    {
        "text": "The courts of the State of California shall have exclusive jurisdiction over any proceeding arising hereunder.",
        "category": "GOVERNING_LAW",
    },
    # Indemnification clauses
    {
        "text": "The contractor shall indemnify and hold harmless the client against all liabilities claims losses and damages.",
        "category": "INDEMNIFICATION",
    },
    {
        "text": "Each party agrees to defend and indemnify the other party from any third party intellectual property infringement claims.",
        "category": "INDEMNIFICATION",
    },
    {
        "text": "The vendor agrees to indemnify the purchaser for any losses arising from negligence or willful misconduct.",
        "category": "INDEMNIFICATION",
    },
    {
        "text": "The company shall not be liable for indirect special incidental or consequential damages arising hereunder.",
        "category": "INDEMNIFICATION",
    },
    {
        "text": "The service provider shall hold harmless the customer from any regulatory penalties caused by security breach.",
        "category": "INDEMNIFICATION",
    },
    # Payment clauses
    {
        "text": "The client shall pay all invoiced fees within thirty days of the invoice date.",
        "category": "PAYMENT",
    },
    {
        "text": "All payments due under this agreement shall be made in United States dollars via wire transfer.",
        "category": "PAYMENT",
    },
    {
        "text": "Late payments shall accrue interest at the rate of one and one half percent per month.",
        "category": "PAYMENT",
    },
    {
        "text": "The service fee shall be payable monthly in advance on the first business day of each calendar month.",
        "category": "PAYMENT",
    },
    {
        "text": "In consideration for the services provided the company shall compensate the consultant at the agreed hourly rate.",
        "category": "PAYMENT",
    },
]


# =====================================================================
# 1. N-GRAM LANGUAGE MODEL (TOKENIZATION, PROBABILITIES, SMOOTHING)
# =====================================================================

def tokenize_legal_text(text: str) -> list[str]:
    """Tokenize legal text into normalized lowercase word tokens."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in cleaned.split() if t]


class NGramPredictor:
    """N-gram next-word language model with Add-alpha smoothing and backoff."""

    def __init__(self, n: int = 3, alpha: float = 0.05):
        self.n = max(2, n)
        self.alpha = alpha
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_totals: dict[tuple[str, ...], int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.sentences: list[list[str]] = []

    def fit(self, texts: list[str]) -> NGramPredictor:
        """Train n-gram model counts from a list of legal sentences."""
        self.counts.clear()
        self.context_totals.clear()
        self.vocab.clear()
        self.sentences.clear()

        for text in texts:
            tokens = tokenize_legal_text(text)
            if not tokens:
                continue
            self.sentences.append(tokens)
            self.vocab.update(tokens)

            # Store counts for all orders from 1 up to self.n
            for order in range(1, self.n + 1):
                padding = ["<bos>"] * (order - 1)
                padded_tokens = padding + tokens + ["<eos>"]
                for i in range(len(padded_tokens) - order + 1):
                    ngram = tuple(padded_tokens[i : i + order])
                    ctx, word = ngram[:-1], ngram[-1]
                    self.counts[ctx][word] += 1
                    self.context_totals[ctx] += 1

        self.vocab.add("<eos>")
        return self

    def next_word_distribution(self, context_tokens: list[str]) -> dict[str, float]:
        """Compute smoothed next-word probability distribution given recent context."""
        v_size = max(1, len(self.vocab))

        # Backoff from order-1 down to unigram
        for order in range(self.n - 1, -1, -1):
            if order == 0:
                ctx = ()
            else:
                needed = context_tokens[-(order):] if len(context_tokens) >= order else (["<bos>"] * (order - len(context_tokens)) + context_tokens)
                ctx = tuple(needed)

            total = self.context_totals.get(ctx, 0)
            if total > 0 or order == 0:
                word_counts = self.counts.get(ctx, Counter())
                denom = total + self.alpha * v_size
                probs = {}
                for w in self.vocab:
                    if w == "<bos>":
                        continue
                    prob = (word_counts.get(w, 0) + self.alpha) / denom
                    probs[w] = prob
                total_p = sum(probs.values()) or 1.0
                return {w: p / total_p for w, p in probs.items()}

        # Fallback uniform over vocabulary
        uniform = 1.0 / v_size
        return {w: uniform for w in self.vocab if w != "<bos>"}

    def predict_next(self, prompt: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-K most likely next words and their probabilities."""
        tokens = tokenize_legal_text(prompt)
        dist = self.next_word_distribution(tokens)
        sorted_candidates = sorted(dist.items(), key=lambda x: -x[1])
        return sorted_candidates[:top_k]

    def generate(self, prompt: str, max_words: int = 12) -> str:
        """Autoregressively extend a prompt until <eos> or max_words reached."""
        tokens = tokenize_legal_text(prompt)
        generated = list(tokens)

        for _ in range(max_words):
            dist = self.next_word_distribution(generated)
            # Greedy argmax next word
            best_word = max(dist.items(), key=lambda x: x[1])[0]
            if best_word == "<eos>":
                break
            generated.append(best_word)

        return " ".join(generated)


# =====================================================================
# 2. GPT-2 CLAUSE CLASSIFIER (PROMPT LOSS MASKING & CONSTRAINED DECODING)
# =====================================================================

class LegalClauseClassifier:
    """GPT-2 generative legal clause classifier using constrained label decoding."""

    def __init__(self, model: Optional[GPT2LMHeadModel] = None, tokenizer: Optional[PreTrainedTokenizer] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.label_token_ids: dict[str, int] = {}
        if self.tokenizer:
            self._setup_labels()

    def _setup_labels(self) -> None:
        """Register custom atomic label tokens and map their token IDs."""
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": LABEL_TOKENS})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model and added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.label_token_ids = {
            tok: self.tokenizer.convert_tokens_to_ids(tok)
            for tok in LABEL_TOKENS
        }

    @classmethod
    def create_model(cls, use_pretrained: bool = True) -> LegalClauseClassifier:
        """Instantiate classifier with GPT-2 tokenizer and model."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_pretrained:
            try:
                model = GPT2LMHeadModel.from_pretrained("gpt2")
            except Exception:
                # Lightweight CPU-safe fallback config matching GPT-2 architecture
                config = GPT2Config(
                    vocab_size=len(tokenizer) + len(LABEL_TOKENS) + 10,
                    n_positions=256,
                    n_embd=128,
                    n_layer=4,
                    n_head=4,
                )
                model = GPT2LMHeadModel(config)
        else:
            config = GPT2Config(
                vocab_size=len(tokenizer) + len(LABEL_TOKENS) + 10,
                n_positions=256,
                n_embd=128,
                n_layer=4,
                n_head=4,
            )
            model = GPT2LMHeadModel(config)

        classifier = cls(model=model, tokenizer=tokenizer)
        classifier._setup_labels()
        return classifier

    def format_prompt(self, text: str) -> str:
        """Format raw legal clause text into training/inference prompt."""
        return f"Clause: {text.strip()} Type: "

    def train_on_corpus(
        self,
        examples: list[dict[str, str]],
        epochs: int = 5,
        lr: float = 3e-4,
    ) -> list[float]:
        """Fine-tune GPT-2 model using prompt loss masking (-100 on prompt tokens)."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # Prepare formatted sequences with label targets
        items = []
        for ex in examples:
            prompt = self.format_prompt(ex["text"])
            label_tok = CAT_TO_LABEL.get(ex["category"], "[CONFIDENTIALITY]")
            full_text = prompt + label_tok

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

            # Target labels: mask prompt with -100 so loss is only on the label token
            target_ids = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

            items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(target_ids, dtype=torch.long),
            })

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for item in items:
                optimizer.zero_grad()
                input_ids = item["input_ids"].unsqueeze(0)
                labels = item["labels"].unsqueeze(0)
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, len(items)))

        self.model.eval()
        return losses

    def classify(self, text: str) -> dict[str, Any]:
        """Classify a legal clause via constrained argmax decoding over label tokens."""
        self.model.eval()
        prompt = self.format_prompt(text)
        input_ids = torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            # Logits at the final prediction position
            next_token_logits = outputs.logits[0, -1, :]

            # Slice logits restricted exclusively to valid label tokens
            restricted_logits = []
            valid_labels = []
            for tok in LABEL_TOKENS:
                tok_id = self.label_token_ids[tok]
                restricted_logits.append(next_token_logits[tok_id].item())
                valid_labels.append(tok)

            # Softmax over constrained label set
            t_logits = torch.tensor(restricted_logits)
            probs = torch.softmax(t_logits, dim=0).tolist()

        best_idx = int(torch.argmax(t_logits).item())
        best_label = valid_labels[best_idx]
        category = LABEL_TO_CAT[best_label]
        confidence = probs[best_idx]

        breakdown = {LABEL_TO_CAT[tok]: round(p, 4) for tok, p in zip(valid_labels, probs)}

        return {
            "category": category,
            "label_token": best_label,
            "confidence": round(confidence, 4),
            "probabilities": breakdown,
        }


# =====================================================================
# 3. INTEGRATED ASSISTANT WORKFLOW & EVALUATION
# =====================================================================

class LegalDraftingAssistant:
    """Integrated Legal Drafting Assistant combining N-Gram and GPT-2 models."""

    def __init__(self, predictor: NGramPredictor, classifier: LegalClauseClassifier):
        self.predictor = predictor
        self.classifier = classifier

    def draft_and_classify(self, prompt: str, max_words: int = 10) -> dict[str, Any]:
        """Generate legal sentence continuation and classify the resulting clause."""
        top_suggestions = self.predictor.predict_next(prompt, top_k=5)
        drafted_sentence = self.predictor.generate(prompt, max_words=max_words)
        classification = self.classifier.classify(drafted_sentence)

        return {
            "initial_prompt": prompt,
            "top_next_words": top_suggestions,
            "drafted_sentence": drafted_sentence,
            "predicted_clause_type": classification["category"],
            "confidence": classification["confidence"],
            "class_probabilities": classification["probabilities"],
            "disclaimer": LEGAL_DISCLAIMER,
        }


def evaluate_classifier(
    classifier: LegalClauseClassifier,
    test_data: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute Accuracy, Macro-F1, Precision, Recall, and Confusion Matrix."""
    y_true = [item["category"] for item in test_data]
    y_pred = []

    for item in test_data:
        res = classifier.classify(item["text"])
        y_pred.append(res["category"])

    # Metrics computation
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    total = len(y_true)
    accuracy = correct / max(1, total)

    # Confusion matrix and per-class stats
    categories = sorted(LEGAL_CATEGORIES)
    cm: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in categories} for c in categories}
    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm[yt]:
            cm[yt][yp] += 1

    per_class = {}
    f1_scores = []
    for cat in categories:
        tp = cm[cat][cat]
        fp = sum(cm[other][cat] for other in categories if other != cat)
        fn = sum(cm[cat][other] for other in categories if other != cat)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cat] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(cm[cat].values()),
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / max(1, len(f1_scores))

    return {
        "total_test_samples": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
    }


# =====================================================================
# MILESTONE STUBS (CALLED BY CLI AND TESTS)
# =====================================================================

_CACHED_PREDICTOR: Optional[NGramPredictor] = None
_CACHED_CLASSIFIER: Optional[LegalClauseClassifier] = None


def milestone_1(corpus: Optional[list[dict[str, str]]] = None) -> list[dict[str, str]]:
    """Milestone 1: Prepare a legal text corpus for n-gram training."""
    data = corpus or DEFAULT_LEGAL_CORPUS
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "legal_corpus.json").write_text(json.dumps(data, indent=2))
    return data


def milestone_2(n: int = 3, alpha: float = 0.05) -> NGramPredictor:
    """Milestone 2: Implement and train the n-gram next-word predictor."""
    global _CACHED_PREDICTOR
    corpus = milestone_1()
    texts = [item["text"] for item in corpus]
    predictor = NGramPredictor(n=n, alpha=alpha)
    predictor.fit(texts)
    _CACHED_PREDICTOR = predictor
    return predictor


def milestone_3(use_pretrained: bool = False, epochs: int = 4) -> LegalClauseClassifier:
    """Milestone 3: Load GPT-2 model with custom label tokens and fine-tune on legal clauses."""
    global _CACHED_CLASSIFIER
    classifier = LegalClauseClassifier.create_model(use_pretrained=use_pretrained)
    corpus = milestone_1()
    classifier.train_on_corpus(corpus, epochs=epochs, lr=3e-4)
    _CACHED_CLASSIFIER = classifier
    return classifier


def milestone_4(prompt: str = "The receiving party agrees to", max_words: int = 8) -> dict[str, Any]:
    """Milestone 4: Integrate n-gram text generation with GPT-2 clause classification."""
    global _CACHED_PREDICTOR, _CACHED_CLASSIFIER
    if _CACHED_PREDICTOR is None:
        milestone_2()
    if _CACHED_CLASSIFIER is None:
        milestone_3(use_pretrained=False, epochs=3)

    assistant = LegalDraftingAssistant(_CACHED_PREDICTOR, _CACHED_CLASSIFIER)
    result = assistant.draft_and_classify(prompt, max_words=max_words)
    return result


def milestone_5() -> dict[str, Any]:
    """Milestone 5: Evaluate classification accuracy and metrics on example legal clauses."""
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is None:
        milestone_3(use_pretrained=False, epochs=4)

    test_data = DEFAULT_LEGAL_CORPUS
    eval_results = evaluate_classifier(_CACHED_CLASSIFIER, test_data)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "evaluation_results.json").write_text(json.dumps(eval_results, indent=2))
    return eval_results
