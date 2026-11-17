"""Relation extraction module for the clinical NLP pipeline."""
from src.relation_extraction.relation_engine import (
    RelationEngine, BaseRelationExtractor, RuleBasedRelationExtractor, MLRelationExtractor
)

__all__ = [
    "RelationEngine", "BaseRelationExtractor",
    "RuleBasedRelationExtractor", "MLRelationExtractor",
]
