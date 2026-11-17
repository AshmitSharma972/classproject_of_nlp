"""
Utility Module: Provides logging, custom exceptions, file validation, and formatting helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Union


# Reconfigure stdout/stderr encoding safely on Windows
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# Setup Logger
def get_logger(name: str = "MedicalNLP") -> logging.Logger:
    """Configures and returns a standardized logger safe for Windows Unicode."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger("MedicalNLP.Utils")


# Custom Exception Classes
class MedicalNLPException(Exception):
    """Base exception class for Medical NLP project."""
    pass


class DocumentProcessingError(MedicalNLPException):
    """Raised when document reading or extraction fails."""
    pass


class ScannedDocumentException(DocumentProcessingError):
    """Raised when a PDF contains no extractable text layer (Scanned/Image-only)."""
    pass


class SummarizationError(MedicalNLPException):
    """Raised when text summarization fails."""
    pass


class ExtractionError(MedicalNLPException):
    """Raised when medical NER or information extraction fails."""
    pass


def validate_file_path(file_path: Union[str, Path]) -> Path:
    """Validates that a file path exists and is a readable file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at specified path: {path}")
    if not path.is_file():
        raise ValueError(f"Path exists but is not a file: {path}")
    return path


def clean_text_whitespace(text: str) -> str:
    """Normalizes excessive whitespace, blank lines, and line endings."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)
