"""
Unit tests for InformationExtractor and MedicalNER.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.information_extractor import InformationExtractor


class TestInformationExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = InformationExtractor()

    def test_negation_detection(self):
        sentences = ["Patient has no fever and denies chest pain."]
        res = self.extractor.process_sentences(sentences)

        raw_entities = res["raw_entities"]
        negated_texts = [e["text"].lower() for e in raw_entities if e["context"] == "NEGATED"]

        self.assertIn("fever", negated_texts)
        self.assertIn("chest pain", negated_texts)

    def test_past_history_detection(self):
        sentences = ["Patient had fever last week and history of hypertension."]
        res = self.extractor.process_sentences(sentences)

        raw_entities = res["raw_entities"]
        past_entities = [e["text"].lower() for e in raw_entities if e["context"] == "PAST"]

        self.assertTrue(len(past_entities) > 0)

    def test_structured_slot_populating(self):
        sentences = ["Patient has hypertension and diabetes.", "Currently taking Lisinopril 20 mg."]
        res = self.extractor.process_sentences(sentences)
        slots = res["structured_slots"]

        self.assertTrue(len(slots["Medical History"]) > 0 or len(slots["Symptoms"]) > 0)
        self.assertTrue(len(slots["Medications"]) > 0)


if __name__ == "__main__":
    unittest.main()
