"""
Unit tests for MedicalSummarizer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.summarizer import MedicalSummarizer


class TestMedicalSummarizer(unittest.TestCase):

    def setUp(self):
        self.summarizer = MedicalSummarizer(prefer_transformer=False)

    def test_summary_generation_extractive(self):
        sentences = [
            "Patient presented with acute chest pain and dyspnea.",
            "ECG showed ST elevation consistent with acute MI.",
            "Patient was started on Lisinopril and Aspirin."
        ]
        full_text = " ".join(sentences)
        key_points = [{"sentence": sentences[0], "sentence_idx": 0, "score": 10.0}]

        res = self.summarizer.generate_summary(full_text, sentences, key_points)
        self.assertTrue(len(res["summary_text"]) > 0)
        self.assertIn("mode", res)


if __name__ == "__main__":
    unittest.main()
