"""
Preprocessing sub-package for the Clinical NLP Pipeline.

Public API
----------
ClinicalPreprocessor  - the main entry point for text normalization and
                        sentence segmentation.
PreprocessingOutput   - Pydantic schema (re-exported from common.schemas)
                        that all downstream stages consume.

Usage
-----
    from src.preprocessing import ClinicalPreprocessor, PreprocessingOutput

    preprocessor = ClinicalPreprocessor()
    output: PreprocessingOutput = preprocessor.run(raw_text)
"""

from src.common.schemas import PreprocessingOutput
from src.preprocessing.preprocessor import ClinicalPreprocessor

__all__ = ["ClinicalPreprocessor", "PreprocessingOutput"]
