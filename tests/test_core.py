"""Comprehensive unit tests for Legal Drafting Assistant milestones.

Validates:
1. Legal corpus creation, stratified splitting, and tokenization.
2. N-gram language model estimation, baseline comparison, and generation.
3. GPT-2 clause tagging with custom label tokens and constrained decoding.
4. Integrated interactive drafting workflow.
5. Evaluation metrics, baselines, confusion matrix, and artifact export.
"""
from pathlib import Path
import pytest
import torch

from app import core


def test_milestone_1():
    """Milestone 1: Verify legal text corpus preparation, stratified splitting, and persistence."""
    corpus = core.milestone_1()
    assert len(corpus) == 50
    assert Path("data/legal_corpus.json").exists()

    # Check categories
    found_cats = {item["category"] for item in corpus}
    assert found_cats == set(core.LEGAL_CATEGORIES)

    # Test stratified splitting
    train_data, val_data, test_data = core.split_dataset(corpus, seed=42)
    assert len(train_data) == 30
    assert len(val_data) == 10
    assert len(test_data) == 10

    # Ensure zero data leakage between train and test
    train_texts = {item["text"] for item in train_data}
    test_texts = {item["text"] for item in test_data}
    assert len(train_texts.intersection(test_texts)) == 0

    # Test tokenization helper
    tokens = core.tokenize_legal_text("The Receiving Party SHALL NOT disclose!")
    assert tokens == ["the", "receiving", "party", "shall", "not", "disclose"]


def test_milestone_2():
    """Milestone 2: Verify n-gram next-word predictor counts, smoothing, and baseline comparisons."""
    predictor = core.milestone_2(n=3, alpha=0.1)

    assert len(predictor.vocab) > 20
    assert "<eos>" in predictor.vocab

    # Test next-word distribution
    dist = predictor.next_word_distribution(["receiving", "party"])
    assert isinstance(dist, dict)
    assert len(dist) > 0
    assert abs(sum(dist.values()) - 1.0) < 1e-4

    # Test top-k prediction
    top_candidates = predictor.predict_next("receiving party", top_k=5)
    assert len(top_candidates) == 5
    for word, prob in top_candidates:
        assert isinstance(word, str)
        assert 0.0 <= prob <= 1.0

    # Test continuation generation
    generated = predictor.generate("The receiving party agrees to", max_words=6)
    assert isinstance(generated, str)
    assert len(generated.split()) > len("The receiving party agrees to".split())

    # Graceful handling of completely unseen context
    unseen_dist = predictor.next_word_distribution(["completely", "alien", "context"])
    assert len(unseen_dist) > 0
    assert abs(sum(unseen_dist.values()) - 1.0) < 1e-4

    # Test ngram baseline comparison evaluation
    train_texts = [item["text"] for item in core.EXPANDED_LEGAL_CORPUS[:30]]
    test_texts = [item["text"] for item in core.EXPANDED_LEGAL_CORPUS[30:]]
    ngram_eval = core.evaluate_ngram_models(train_texts, test_texts)
    assert "Unigram (Baseline)" in ngram_eval
    assert "Bigram" in ngram_eval
    assert "Trigram (Proposed)" in ngram_eval


def test_milestone_3():
    """Milestone 3: Verify GPT-2 model loading, label expansion, and constrained decoding."""
    classifier = core.LegalClauseClassifier.create_model(use_pretrained=False)

    # Verify custom atomic label tokens are registered in tokenizer
    for label_tok in core.LABEL_TOKENS:
        assert label_tok in classifier.label_token_ids
        tok_id = classifier.label_token_ids[label_tok]
        assert tok_id < classifier.model.config.vocab_size

    # Verify model embeddings were properly resized
    assert classifier.model.get_input_embeddings().weight.shape[0] == len(classifier.tokenizer)

    # Verify prompt formatting
    prompt = classifier.format_prompt("This is a test clause.")
    assert prompt == "Clause: This is a test clause. Type: "

    # Verify constrained classification inference
    res = classifier.classify("Either party may terminate this contract upon written notice.")
    assert "category" in res
    assert res["category"] in core.LEGAL_CATEGORIES
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert len(res["probabilities"]) == len(core.LEGAL_CATEGORIES)


def test_milestone_4():
    """Milestone 4: Verify integrated drafting assistant and legal disclaimer."""
    result = core.milestone_4(prompt="The receiving party agrees to", max_words=6)

    assert "initial_prompt" in result
    assert "top_next_words" in result
    assert len(result["top_next_words"]) <= 5
    assert "drafted_sentence" in result
    assert "predicted_clause_type" in result
    assert result["predicted_clause_type"] in core.LEGAL_CATEGORIES
    assert "confidence" in result
    assert "disclaimer" in result
    assert "educational AI demonstration" in result["disclaimer"]


def test_milestone_5():
    """Milestone 5: Verify evaluation metrics, baseline comparisons, and output artifact exports."""
    eval_summary = core.milestone_5()

    assert "dataset" in eval_summary
    assert "gpt2_classification_comparison" in eval_summary
    assert "ngram_language_model_comparison" in eval_summary

    gpt2_comp = eval_summary["gpt2_classification_comparison"]
    assert "Fine-Tuned GPT-2 (Proposed)" in gpt2_comp
    assert "Pretrained Zero-Shot GPT-2 (Baseline)" in gpt2_comp
    assert "Majority-Class (Baseline)" in gpt2_comp

    # Verify generated output files exist in outputs/
    outputs_dir = Path("outputs")
    assert (outputs_dir / "evaluation_results.json").exists()
    assert (outputs_dir / "sample_predictions.json").exists()
    assert (outputs_dir / "confusion_matrix.csv").exists()
    assert (outputs_dir / "ngram_prediction_examples.json").exists()
    assert (outputs_dir / "evaluation_report.md").exists()
