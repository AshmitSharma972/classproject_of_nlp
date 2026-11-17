"""Evaluate a local mT5 checkpoint on the held-out synthetic summary split.

ROUGE is lexical overlap, not clinical accuracy or clinical validation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rouge_score import rouge_scorer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def score_summaries(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    totals = {f"{name}_{metric}": 0.0
              for name in ("rouge1", "rouge2", "rougeL")
              for metric in ("precision", "recall", "f1")}
    if not predictions:
        return totals
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for name in ("rouge1", "rouge2", "rougeL"):
            totals[f"{name}_precision"] += scores[name].precision
            totals[f"{name}_recall"] += scores[name].recall
            totals[f"{name}_f1"] += scores[name].fmeasure
    return {name: round(value / len(predictions) * 100, 2) for name, value in totals.items()}


def evaluate(model_path: Path, data_path: Path, target: str = "summary_en", max_samples: int = 100,
             batch_size: int = 4) -> dict:
    if max_samples < 1 or batch_size < 1:
        raise ValueError("max_samples and batch_size must be positive")
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    first_split = dataset.train_test_split(test_size=0.10, seed=42)
    validation_test = first_split["test"].train_test_split(test_size=0.50, seed=42)
    held_out = validation_test["test"]
    held_out = held_out.filter(lambda row: isinstance(row.get(target), str) and bool(row[target].strip()))
    rows = held_out.select(range(min(max_samples, len(held_out))))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    predictions, references, languages = [], [], []
    with torch.inference_mode():
        batches = range(0, len(rows), batch_size)
        for start in tqdm(batches, total=(len(rows) + batch_size - 1) // batch_size,
                          desc="Generating held-out summaries", unit="batch"):
            batch = rows[start:start + batch_size]
            inputs = tokenizer(batch["source_text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generated = model.generate(**inputs, num_beams=2, max_new_tokens=128)
            predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            references.extend(batch[target])
            languages.extend(batch.get("language", ["unknown"] * len(batch[target])))

    by_language = defaultdict(lambda: {"predictions": [], "references": []})
    for language, prediction, reference in zip(languages, predictions, references):
        bucket = by_language[str(language)]
        bucket["predictions"].append(prediction); bucket["references"].append(reference)
    result = {
        "model_path": str(model_path), "dataset": str(data_path), "target": target,
        "split": "same seeded 5% test split as training script", "samples_evaluated": len(predictions),
        "batch_size": batch_size, "device": str(device),
        "overall": score_summaries(predictions, references),
        "per_language": {language: {"samples": len(values["predictions"]),
                                    **score_summaries(values["predictions"], values["references"])}
                         for language, values in sorted(by_language.items())},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "notice": "Synthetic held-out ROUGE scores measure text overlap only; they are not clinical accuracy or clinical validation.",
    }
    output_path = model_path / "summary_evaluation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "clinical_transformer_weights")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "synthetic_multilingual_clinical_cases.jsonl")
    parser.add_argument("--target", choices=("summary", "summary_en"), default="summary_en")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.data, args.target, args.max_samples, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
