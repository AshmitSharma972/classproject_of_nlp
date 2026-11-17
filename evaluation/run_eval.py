"""Score only clinician-approved cases; TODO/synthetic slots are excluded."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

GOLD = Path(__file__).parent / "gold" / "clinical_cases.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def _norm(value):
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _prf(gold, predicted):
    g, p = set(gold), set(predicted)
    tp = len(g & p); precision = tp / len(p) if p else 0.0; recall = tp / len(g) if g else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "gold": len(g), "predicted": len(p)}


def _seeded_translation_checks():
    seeds = [
        ("No fever for 5 days", "Fever for 5 days", "no"),
        ("Metformin 500 mg twice daily", "Metformin twice daily", "500 mg"),
        ("Cough since yesterday", "Cough since last week", "yesterday"),
    ]
    detected = 0
    for source, translation, critical in seeds:
        if critical.casefold() not in translation.casefold():
            detected += 1
    return {"seeded_errors": len(seeds), "detected": detected,
            "notice": "Seeded heuristic checks only; not a calibrated translation-quality score."}


def _summary_model_status():
    model_root = PROJECT_ROOT / "models"
    candidates = sorted({
        model_root / "mt5-clinical-synthetic",
        model_root / "clinical_transformer_weights",
        model_root / "mt5-clinical-native",
        *(path for path in model_root.iterdir() if path.is_dir()),
    }) if model_root.is_dir() else []
    checkpoints = []
    training_metrics = None
    summary_evaluation = None
    training_status = "No local summary-model checkpoint found."
    for directory in candidates:
        if not directory.is_dir():
            continue
        weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
        if not weights:
            continue
        total_bytes = sum(file.stat().st_size for file in weights)
        checkpoints.append({"name": directory.name, "size_bytes": total_bytes,
                            "weight_files": len(weights)})
        metrics_file = directory / "training_metrics.json"
        state_file = directory / "trainer_state.json"
        if metrics_file.is_file():
            try:
                training_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        summary_file = directory / "summary_evaluation.json"
        if summary_file.is_file():
            try:
                summary_evaluation = json.loads(summary_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if training_metrics is None and state_file.is_file():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                history = state.get("log_history", [])
                evaluation = next((row for row in reversed(history) if any(key.startswith("eval_") for key in row)), None)
                if evaluation:
                    training_metrics = {"test_metrics": {key: value for key, value in evaluation.items() if key.startswith("eval_")},
                                        "source": "trainer_state.json", "notice": "Training loss/metrics only; not clinical accuracy."}
            except (OSError, json.JSONDecodeError):
                pass
    if checkpoints:
        training_status = ("Training metrics recorded." if training_metrics else
                           "Local weights found; fine-tuning status and training metrics are not recorded.")
    return {"active_path": "template", "active_in_live_pipeline": False,
            "training_status": training_status, "checkpoints": checkpoints,
            "training_metrics": training_metrics, "summary_evaluation": summary_evaluation,
            "notice": "The live clinical summary uses templates. Local mT5 weights are not used by the live summary path."}


def run():
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    approved = [r for r in rows if r.get("review_status") == "clinician_approved" and isinstance(r.get("annotations"), dict)]
    groups = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)
    pipeline = None
    reports = {}
    for language, cases in groups.items():
        usable = [r for r in approved if r["language"] == language and
                  (language == "English" or isinstance(r.get("english_text"), str) and r["english_text"].strip())]
        if not usable:
            reports[language] = {"approved_cases": 0, "available_cases": len(cases), "metrics": None,
                                 "status": "No clinician-approved cases; metrics not reported."}
            continue
        if pipeline is None:
            from src.pipeline import ClinicalPipeline
            pipeline = ClinicalPipeline()
        entity_gold, entity_pred, assertion_correct, assertion_total = [], [], 0, 0
        temporal_correct = temporal_total = medication_correct = medication_total = 0
        for row in usable:
            result = pipeline.run(row["text"] if language == "English" else row["english_text"])
            ann = row["annotations"]
            entity_gold.extend((_norm(x["text"]), x["type"]) for x in ann.get("entities", []))
            entity_pred.extend((_norm(e.span.text), e.type.value) for e in result.entities)
            gold_assert = {_norm(x["text"]): x["status"] for x in ann.get("assertions", [])}
            pred_assert = {_norm(e.span.text): a.status.value for e in result.entities
                           for a in result.assertions if a.entity_id == e.id}
            assertion_total += len(gold_assert)
            assertion_correct += sum(pred_assert.get(k) == v for k, v in gold_assert.items())
            pred_t = {(_norm(next((e.span.text for e in result.entities if e.id == eid), "")), _norm(t.normalized_value or t.raw_value))
                      for t in result.time_expressions for eid in t.linked_entity_ids}
            expected_t = {(_norm(x["entity"]), _norm(x["time"])) for x in ann.get("temporal_links", [])}
            temporal_total += len(expected_t); temporal_correct += len(expected_t & pred_t)
            pred_m = {(_norm(next((e.span.text for e in result.entities if e.id == rel.head_entity_id), "")),
                         rel.relation_type.value, _norm(next((e.span.text for e in result.entities if e.id == rel.tail_entity_id), "")))
                      for rel in result.relations if rel.relation_type.value.startswith("MEDICATION-")}
            expected_m = {(_norm(x["drug"]), x["relation"], _norm(x["value"])) for x in ann.get("medication_relations", [])}
            medication_total += len(expected_m); medication_correct += len(expected_m & pred_m)
        reports[language] = {"approved_cases": len(usable), "metrics": {
            "entity_precision_recall_f1": _prf(entity_gold, entity_pred),
            "assertion_accuracy": assertion_correct / assertion_total if assertion_total else None,
            "temporal_link_accuracy": temporal_correct / temporal_total if temporal_total else None,
            "medication_relation_accuracy": medication_correct / medication_total if medication_total else None}}
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
            "per_language": reports, "clinical_pipeline_metrics": reports,
            "summary_model": _summary_model_status(),
            "translation_critical_token_detection": _seeded_translation_checks(),
            "notice": "All available gold slots are synthetic and currently unapproved. These metrics do not establish clinical accuracy or validity."}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
