"""Fine-tune mT5 on the synthetic multilingual clinical-note corpus.

For research/education only. Do not use the resulting model for diagnosis or
care decisions without appropriate clinical validation and governance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_multilingual_clinical_cases.jsonl"))
    parser.add_argument("--output-dir", default="models/mt5-clinical-synthetic")
    parser.add_argument("--model", default="google/mt5-small")
    parser.add_argument("--target", choices=("summary", "summary_en"), default="summary_en")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    data = load_dataset("json", data_files=str(args.data), split="train")
    split = data.train_test_split(test_size=0.10, seed=42)
    validation_test = split["test"].train_test_split(test_size=0.50, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def tokenize(batch):
        model_inputs = tokenizer(batch["source_text"], max_length=512, truncation=True)
        labels = tokenizer(text_target=batch[args.target], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = {
        "train": split["train"].map(tokenize, batched=True, remove_columns=split["train"].column_names),
        "validation": validation_test["train"].map(tokenize, batched=True, remove_columns=validation_test["train"].column_names),
        "test": validation_test["test"].map(tokenize, batched=True, remove_columns=validation_test["test"].column_names),
    }
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_metrics("test", test_metrics)
    output_dir = Path(args.output_dir)
    training_record = {
        "model": args.model,
        "target": args.target,
        "dataset": str(args.data),
        "train_cases": len(tokenized["train"]),
        "validation_cases": len(tokenized["validation"]),
        "test_cases": len(tokenized["test"]),
        "epochs": args.epochs,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "test_metrics": test_metrics,
        "notice": "Synthetic-data training metrics only. Test loss is not clinical accuracy or evidence of clinical validity.",
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(training_record, indent=2), encoding="utf-8")
    print(json.dumps(training_record, indent=2))


if __name__ == "__main__":
    main()
