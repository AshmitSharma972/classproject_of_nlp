"""
src/ner/__init__.py
====================

Public API for the clinical NER module.

Exports
-------
ClinicalNER : The main rule-based NER engine.
BaseNER      : Abstract base class for all NER adapters.
HFNERAdapter : Stub for a HuggingFace-backed NER engine.
"""

from src.ner.ner_engine import BaseNER, ClinicalNER, HFNERAdapter

__all__ = ["ClinicalNER", "BaseNER", "HFNERAdapter"]
