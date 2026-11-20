"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
import sys

from app import core


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="legal-drafting-assistant",
        description="Paralegals get an assistant to draft simple legal sentences using n-gram prediction and instantly classify them into clause types with a fine-tuned GPT-2 model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Base verification command
    sub.add_parser("hello", help="check that everything is wired up")

    # Milestone 1: Prepare legal corpus
    corp_p = sub.add_parser("corpus", help="Prepare and inspect the legal text corpus")
    corp_p.add_argument("--show-samples", action="store_true", help="Display sample clauses per category")

    # Milestone 2: N-Gram Next-Word Prediction
    pred_p = sub.add_parser("predict", help="Predict next words using trained n-gram model")
    pred_p.add_argument("prompt", type=str, help="Initial legal phrase or sentence starter")
    pred_p.add_argument("--top-k", type=int, default=5, help="Number of next-word candidates to show (default: 5)")
    pred_p.add_argument("--generate", action="store_true", help="Generate full sentence continuation")
    pred_p.add_argument("--max-words", type=int, default=10, help="Maximum words to generate (default: 10)")

    # Milestone 3: GPT-2 Clause Classification
    cls_p = sub.add_parser("classify", help="Classify a legal clause into standard categories using fine-tuned GPT-2")
    cls_p.add_argument("clause", type=str, help="Text of the legal clause to classify")

    # Milestone 4: Integrated Drafting Assistant
    draft_p = sub.add_parser("draft", help="Interactively draft a sentence with n-gram suggestions and classify it with GPT-2")
    draft_p.add_argument("prompt", type=str, help="Initial legal draft phrase")
    draft_p.add_argument("--max-words", type=int, default=10, help="Maximum continuation length")

    # Milestone 5: Evaluation
    eval_p = sub.add_parser("evaluate", help="Evaluate GPT-2 classification accuracy and confusion matrix")
    eval_p.add_argument("--output", type=str, default=None, help="Optional custom JSON output path")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Legal Drafting Assistant is ready. Open README.md for your milestones.")
    elif args.command == "corpus":
        data = core.milestone_1()
        print(f"Legal corpus prepared with {len(data)} clauses across {len(core.LEGAL_CATEGORIES)} categories:")
        for cat in core.LEGAL_CATEGORIES:
            count = sum(1 for c in data if c["category"] == cat)
            print(f"  - {cat:16s}: {count} clauses")
        if args.show_samples:
            print("\nSample Clauses:")
            for cat in core.LEGAL_CATEGORIES:
                sample = next(c["text"] for c in data if c["category"] == cat)
                print(f"  [{cat}] {sample}")
    elif args.command == "predict":
        predictor = core.milestone_2()
        candidates = predictor.predict_next(args.prompt, top_k=args.top_k)
        print(f"Prompt: \"{args.prompt}\"")
        print(f"Top {len(candidates)} Next-Word Candidates:")
        for idx, (word, prob) in enumerate(candidates, 1):
            print(f"  [{idx}] '{word}': {prob * 100:5.2f}%")
        if args.generate:
            continuation = predictor.generate(args.prompt, max_words=args.max_words)
            print(f"\nGenerated Continuation: \"{continuation}\"")
    elif args.command == "classify":
        classifier = core.milestone_3(use_pretrained=False, epochs=4)
        result = classifier.classify(args.clause)
        print(f"Clause: \"{args.clause}\"")
        print(f"Predicted Category: [{result['category']}] (Confidence: {result['confidence'] * 100:.1f}%)")
        print("Class Probability Breakdown:")
        for cat, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {cat:16s}: {p * 100:5.2f}%")
        print(f"\nDisclaimer: {core.LEGAL_DISCLAIMER}")
    elif args.command == "draft":
        result = core.milestone_4(prompt=args.prompt, max_words=args.max_words)
        print("=" * 75)
        print("LEGAL DRAFTING ASSISTANT")
        print("=" * 75)
        print(f"Initial Prompt:   \"{result['initial_prompt']}\"")
        print("\nTop Next-Word Suggestions:")
        for idx, (w, p) in enumerate(result["top_next_words"], 1):
            print(f"  ({idx}) {w:15s} (p={p * 100:4.1f}%)")
        print(f"\nRefined Draft:    \"{result['drafted_sentence']}\"")
        print(f"Predicted Clause: [{result['predicted_clause_type']}] (Confidence: {result['confidence'] * 100:.1f}%)")
        print("\nProbability Distribution:")
        for cat, p in sorted(result["class_probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {cat:16s}: {p * 100:5.2f}%")
        print("-" * 75)
        print(f"Disclaimer: {result['disclaimer']}")
        print("=" * 75)
    elif args.command == "evaluate":
        results = core.milestone_5()
        print("=" * 75)
        print("LEGAL CLAUSE CLASSIFICATION EVALUATION REPORT")
        print("=" * 75)
        print(f"Total Test Samples: {results['total_test_samples']}")
        print(f"Overall Accuracy:   {results['accuracy'] * 100:.2f}%")
        print(f"Macro-F1 Score:     {results['macro_f1'] * 100:.2f}%")
        print("\nPer-Class Breakdown:")
        print(f"{'Category':<18} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        print("-" * 65)
        for cat, m in results["per_class_metrics"].items():
            print(f"{cat:<18} {m['precision'] * 100:6.1f}%      {m['recall'] * 100:6.1f}%      {m['f1'] * 100:6.1f}%      {m['support']}")
        print("-" * 65)
        print("\nConfusion Matrix (Rows = True, Cols = Predicted):")
        cats = sorted(core.LEGAL_CATEGORIES)
        header = f"{'':<16} " + " ".join(f"{c[:4]:>6}" for c in cats)
        print(header)
        for c1 in cats:
            row_str = f"{c1:<16} " + " ".join(f"{results['confusion_matrix'][c1][c2]:>6d}" for c2 in cats)
            print(row_str)
        print("=" * 75)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
