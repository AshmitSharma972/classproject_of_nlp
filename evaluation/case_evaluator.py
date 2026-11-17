"""
Clinical Case Evaluation Module.

Evaluates the 9-stage clinical NLP pipeline against the synthetic ground truth
benchmark in data/sample/ground_truth.json.

Computes Precision, Recall, and F1 for:
1. Named Entity Recognition (NER)
2. Assertion / Negation Detection
3. Relation Extraction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.pipeline import ClinicalPipeline, PipelineResult
from src.utils import get_logger

logger = get_logger("CaseEvaluator")


def _calc_prf1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate precision, recall, f1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


class CaseEvaluator:
    """Evaluates ClinicalPipeline outputs against data/sample/ground_truth.json."""

    def __init__(self, data_dir: Path = Path("data/sample")):
        self.data_dir = data_dir
        self.gt_file = data_dir / "ground_truth.json"
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict[str, Any]:
        if not self.gt_file.exists():
            logger.warning("Ground truth file %s not found.", self.gt_file)
            return {}
        try:
            with open(self.gt_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load ground truth: %s", e)
            return {}

    def evaluate_all(self) -> Dict[str, Any]:
        """Run evaluation over all cases in ground_truth.json."""
        if not self.ground_truth:
            return {"error": "No ground truth available"}

        pipeline = ClinicalPipeline()

        total_ner_tp = total_ner_fp = total_ner_fn = 0
        total_ass_tp = total_ass_fp = total_ass_fn = 0
        total_rel_tp = total_rel_fp = total_rel_fn = 0

        case_results = []

        for case_key, gt_data in self.ground_truth.items():
            case_file = self.data_dir / f"{case_key}.txt"
            if not case_file.exists():
                continue

            raw_text = case_file.read_text(encoding="utf-8")
            res: PipelineResult = pipeline.run(raw_text)

            # --- 1. Evaluate NER ---
            gt_entities = gt_data.get("entities", [])
            pred_entities = res.entities

            pred_set = set(e.span.text.lower() for e in pred_entities)
            gt_set = set(g["text"].lower() for g in gt_entities)

            ner_tp = 0
            for p in pred_set:
                if any(p in g or g in p for g in gt_set):
                    ner_tp += 1
            ner_fp = max(0, len(pred_set) - ner_tp)
            ner_fn = max(0, len(gt_set) - ner_tp)

            total_ner_tp += ner_tp
            total_ner_fp += ner_fp
            total_ner_fn += ner_fn

            # --- 2. Evaluate Assertion ---
            gt_assertions = gt_data.get("assertions", [])
            ent_map = {e.id: e.span.text.lower() for e in res.entities}

            # Build prediction map: text -> status
            pred_ass_map = {}
            for a in res.assertions:
                txt = ent_map.get(a.entity_id, "")
                if txt:
                    pred_ass_map[txt] = a.status.value

            ass_tp = 0
            for g_ass in gt_assertions:
                g_txt = g_ass["entity_text"].lower()
                g_stat = g_ass["status"].upper()
                for p_txt, p_stat in pred_ass_map.items():
                    if (g_txt in p_txt or p_txt in g_txt) and (g_stat == p_stat or (g_stat.startswith("FAMILY") and p_stat == "FAMILY")):
                        ass_tp += 1
                        break

            ass_fp = max(0, len(pred_ass_map) - ass_tp)
            ass_fn = max(0, len(gt_assertions) - ass_tp)

            total_ass_tp += ass_tp
            total_ass_fp += ass_fp
            total_ass_fn += ass_fn

            # --- 3. Evaluate Relation Extraction ---
            gt_relations = gt_data.get("relations", [])
            pred_rel_tuples = []
            for r in res.relations:
                h = ent_map.get(r.head_entity_id, "")
                t = ent_map.get(r.tail_entity_id, "")
                pred_rel_tuples.append((h, t, r.relation_type.value))

            rel_tp = 0
            for g_rel in gt_relations:
                gh = g_rel["head"].lower()
                gt = g_rel["tail"].lower()
                grt = g_rel["type"]
                for ph, pt, prt in pred_rel_tuples:
                    if (gh in ph or ph in gh) and (gt in pt or pt in gt):
                        rel_tp += 1
                        break

            rel_fp = max(0, len(pred_rel_tuples) - rel_tp)
            rel_fn = max(0, len(gt_relations) - rel_tp)

            total_rel_tp += rel_tp
            total_rel_fp += rel_fp
            total_rel_fn += rel_fn

            case_results.append({
                "case": case_key,
                "ner": _calc_prf1(ner_tp, ner_fp, ner_fn),
                "assertion": _calc_prf1(ass_tp, ass_fp, ass_fn),
                "relation": _calc_prf1(rel_tp, rel_fp, rel_fn),
            })

        overall = {
            "ner": _calc_prf1(total_ner_tp, total_ner_fp, total_ner_fn),
            "assertion": _calc_prf1(total_ass_tp, total_ass_fp, total_ass_fn),
            "relation": _calc_prf1(total_rel_tp, total_rel_fp, total_rel_fn),
            "total_cases_evaluated": len(case_results),
            "case_results": case_results,
        }
        return overall

    def print_benchmark_report(self, results: Dict[str, Any]):
        """Format and print benchmark results as a table."""
        sep = "=" * 80
        subsep = "-" * 80
        print("\n" + sep)
        print("CLINICAL NLP PIPELINE BENCHMARK EVALUATION (SAMPLE DATASET)")
        print(sep)
        print(f"{'Case':<10} | {'NER F1':<10} | {'Assertion F1':<14} | {'Relation F1':<12}")
        print(subsep)
        for r in results.get("case_results", []):
            print(f"{r['case']:<10} | {r['ner']['f1']:<10.4f} | {r['assertion']['f1']:<14.4f} | {r['relation']['f1']:<12.4f}")
        print(subsep)
        o_ner = results.get("ner", {})
        o_ass = results.get("assertion", {})
        o_rel = results.get("relation", {})
        print(f"{'OVERALL':<10} | {o_ner.get('f1', 0.0):<10.4f} | {o_ass.get('f1', 0.0):<14.4f} | {o_rel.get('f1', 0.0):<12.4f}")
        print(sep)
        print("\nDETAILED OVERALL METRICS:")
        print(f"NER        : Precision = {o_ner.get('precision', 0):.4f}, Recall = {o_ner.get('recall', 0):.4f}, F1 = {o_ner.get('f1', 0):.4f}")
        print(f"Assertion  : Precision = {o_ass.get('precision', 0):.4f}, Recall = {o_ass.get('recall', 0):.4f}, F1 = {o_ass.get('f1', 0):.4f}")
        print(f"Relation   : Precision = {o_rel.get('precision', 0):.4f}, Recall = {o_rel.get('recall', 0):.4f}, F1 = {o_rel.get('f1', 0):.4f}")
        print(sep + "\n")
