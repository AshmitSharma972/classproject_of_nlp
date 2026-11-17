"""
Relation Extraction Engine for the clinical NLP pipeline.

Consumes NER entities, assertions, and temporal expressions and produces
typed, confidence-scored relations between entity pairs.

Architecture:
  BaseRelationExtractor  (ABC)
  ├── RuleBasedRelationExtractor   ← default, deterministic
  └── MLRelationExtractor          ← stub for future ML/model-based extractor

INVARIANTS:
- Relations are only created for entity pairs within the same sentence.
- Every relation carries the source_sentence_id and optionally the
  evidence_text (the raw sentence text).
- No orchestration logic here -- only relation extraction.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import List

from src.common.schemas import (
    Assertion, Entity, EntityType, PreprocessingOutput, Relation, RelationType, TimeExpr,
)
from src.relation_extraction.rules import ALL_RULES, generate_candidate_pairs
from src.utils import get_logger

logger = get_logger("RelationEngine")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRelationExtractor(ABC):
    """Abstract base for all relation extractors."""

    @abstractmethod
    def extract(
        self,
        entities: List[Entity],
        assertions: List[Assertion],
        time_expressions: List[TimeExpr],
        preprocessing_output: PreprocessingOutput,
    ) -> List[Relation]:
        ...


# ---------------------------------------------------------------------------
# Rule-based extractor (default)
# ---------------------------------------------------------------------------

class RuleBasedRelationExtractor(BaseRelationExtractor):
    """
    Generates candidate entity pairs within the same sentence, then applies
    a priority-ordered list of typed relation rules.

    Confidence: 1.0 for all rule-based extractions (deterministic).
    """

    def extract(
        self,
        entities: List[Entity],
        assertions: List[Assertion],
        time_expressions: List[TimeExpr],
        preprocessing_output: PreprocessingOutput,
    ) -> List[Relation]:
        # Build sentence text lookup for evidence
        sent_text_map = {s.sentence_id: s.text for s in preprocessing_output.sentences}

        candidates = generate_candidate_pairs(entities)
        logger.info("RelationEngine: %d candidate entity pairs generated", len(candidates))

        relations: List[Relation] = []
        seen: set = set()

        # For dosage/frequency values, only the nearest preceding medication
        # in the same clause is a valid source. This prevents all-pairs
        # candidate generation from assigning one value to every drug.
        meds = sorted((e for e in entities if e.type == EntityType.MEDICATION),
                      key=lambda e: (e.span.source_sentence_id, e.span.start))
        value_entities = [e for e in entities if e.type in (EntityType.DOSAGE, EntityType.FREQUENCY)]
        nearest_med_for_value = {}
        for value in value_entities:
            earlier = [m for m in meds if m.span.source_sentence_id == value.span.source_sentence_id
                       and m.span.end <= value.span.start]
            if earlier:
                candidate = earlier[-1]
                sentence = next((s for s in preprocessing_output.sentences
                                 if s.sentence_id == value.span.source_sentence_id), None)
                between = preprocessing_output.raw_text[candidate.span.end:value.span.start]
                if sentence and not any(x in between.lower() for x in (";", ".", " but ", " however ", " although ")):
                    nearest_med_for_value[value.id] = candidate.id

        for head, tail in candidates:
            for rule in ALL_RULES:
                rel_type = rule(head, tail)
                if rel_type is None:
                    continue
                if rel_type in (RelationType.MEDICATION_DOSAGE, RelationType.MEDICATION_FREQUENCY):
                    med = head if head.type == EntityType.MEDICATION else tail
                    value = tail if med is head else head
                    if nearest_med_for_value.get(value.id) != med.id:
                        continue

                # Deduplicate: same (head_id, tail_id, rel_type) only once
                key = (head.id, tail.id, rel_type)
                if key in seen:
                    break
                seen.add(key)

                sent_id = head.span.source_sentence_id
                med = head if head.type == EntityType.MEDICATION else tail
                value = tail if med is head else head
                relation = Relation(
                    id=str(uuid.uuid4()),
                    head_entity_id=med.id if rel_type in (RelationType.MEDICATION_DOSAGE, RelationType.MEDICATION_FREQUENCY) else head.id,
                    tail_entity_id=value.id if rel_type in (RelationType.MEDICATION_DOSAGE, RelationType.MEDICATION_FREQUENCY) else tail.id,
                    relation_type=rel_type,
                    confidence=1.0,
                    source_sentence_id=sent_id,
                    evidence_text=sent_text_map.get(sent_id),
                )
                relations.append(relation)
                break  # first matching rule wins

        logger.info("RelationEngine: %d relations extracted", len(relations))
        return relations


# ---------------------------------------------------------------------------
# ML extractor stub
# ---------------------------------------------------------------------------

class MLRelationExtractor(BaseRelationExtractor):
    """
    Stub for a future ML/model-based relation extractor.

    To implement:
    1. Load a fine-tuned token-classification or sequence-classification model
       (e.g. BioBERT fine-tuned on n2c2 2018 relation task).
    2. For each candidate pair, construct an input string like:
         "[HEAD] {head_text} [/HEAD] context [TAIL] {tail_text} [/TAIL]"
    3. Run model inference to predict relation type and confidence score.
    4. Return Relation objects for predictions above a confidence threshold.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        raise NotImplementedError(
            "MLRelationExtractor is not yet implemented. "
            "See the class docstring for integration guidance."
        )

    def extract(self, entities, assertions, time_expressions, preprocessing_output):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class RelationEngine:
    """
    Public facade for the relation extraction stage.

    Uses RuleBasedRelationExtractor by default.
    Pass extractor=MLRelationExtractor(...) to override.
    """

    def __init__(self, extractor: BaseRelationExtractor | None = None):
        self._extractor = extractor or RuleBasedRelationExtractor()

    def run(
        self,
        entities: List[Entity],
        assertions: List[Assertion],
        time_expressions: List[TimeExpr],
        preprocessing_output: PreprocessingOutput,
    ) -> List[Relation]:
        """
        Run relation extraction.

        Returns:
            List of Relation objects, each typed and confidence-scored.
        """
        return self._extractor.extract(
            entities, assertions, time_expressions, preprocessing_output
        )
