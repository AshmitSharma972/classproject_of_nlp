"""
Unit tests for MedicalNLPEvaluator.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from evaluation.evaluator import MedicalNLPEvaluator


class TestMedicalNLPEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = MedicalNLPEvaluator()

    def test_ner_evaluation_metrics(self):
        extracted = [
            {"text": "chest pain", "category": "Symptoms"},
            {"text": "hypertension", "category": "Medical History"},
            {"text": "lisinopril", "category": "Medications"}
        ]
        reference = ["chest pain", "hypertension", "diabetes"]

        res = self.evaluator.evaluate_ner(extracted, reference)
        self.assertGreater(res["precision"], 0.0)
        self.assertGreater(res["recall"], 0.0)
        self.assertGreater(res["f1_score"], 0.0)

    def test_rouge_evaluation_metrics(self):
        cand = "Patient presented with chest pain and hypertension."
        ref = "Patient presented with severe chest pain and known hypertension."

        res = self.evaluator.evaluate_summarization(cand, ref)
        self.assertIn("rouge1_f1", res)
        self.assertIn("rouge2_f1", res)
        self.assertIn("rougeL_f1", res)
        self.assertGreater(res["rouge1_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
