"""
Unit tests for DocumentProcessor.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.document_processor import DocumentProcessor
from src.utils import DocumentProcessingError
from config import SAMPLE_DIR


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DocumentProcessor()

    def test_extract_from_text(self):
        sample_text = "Patient complains of chest pain. Past history of hypertension."
        res = self.processor.extract_from_text(sample_text)
        self.assertIn("chest pain", res["text"])
        self.assertEqual(res["file_type"], "raw_text")

    def test_extract_txt_file(self):
        sample_path = SAMPLE_DIR / "discharge_summary_01.txt"
        if sample_path.exists():
            res = self.processor.extract_from_file(sample_path)
            self.assertIn("PATIENT DISCHARGE SUMMARY", res["text"])
            self.assertEqual(res["file_type"], "txt")

    def test_unsupported_file_extension(self):
        with self.assertRaises(DocumentProcessingError):
            self.processor.extract_from_file("invalid_file.xyz")


if __name__ == "__main__":
    unittest.main()
