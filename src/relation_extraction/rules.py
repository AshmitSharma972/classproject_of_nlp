"""
Relation extraction rules for the clinical NLP pipeline.

Each rule is a predicate on a (head_entity, tail_entity) pair.
Rules are applied in the order defined; the first matching rule wins.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.common.schemas import Entity, EntityType, RelationType


# ---------------------------------------------------------------------------
# Candidate pair generation
# ---------------------------------------------------------------------------

def generate_candidate_pairs(
    entities: List[Entity],
) -> List[Tuple[Entity, Entity]]:
    """
    Generate all ordered entity pairs within the same sentence.

    Returns pairs (head, tail) where head.span.start <= tail.span.start.
    Only pairs within the same sentence_id are returned.
    """
    pairs: List[Tuple[Entity, Entity]] = []
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            head = entities[i]
            tail = entities[j]
            if head.span.source_sentence_id == tail.span.source_sentence_id:
                pairs.append((head, tail))
    return pairs


# ---------------------------------------------------------------------------
# Individual relation rules
# ---------------------------------------------------------------------------

def _types(head: Entity, tail: Entity) -> Tuple[EntityType, EntityType]:
    return head.type, tail.type


def rule_medication_dosage(head: Entity, tail: Entity) -> Optional[RelationType]:
    """MEDICATION + DOSAGE → MEDICATION-DOSAGE"""
    h, t = _types(head, tail)
    if h == EntityType.MEDICATION and t == EntityType.DOSAGE:
        return RelationType.MEDICATION_DOSAGE
    if h == EntityType.DOSAGE and t == EntityType.MEDICATION:
        return RelationType.MEDICATION_DOSAGE
    return None


def rule_medication_frequency(head: Entity, tail: Entity) -> Optional[RelationType]:
    """MEDICATION + FREQUENCY → MEDICATION-FREQUENCY"""
    h, t = _types(head, tail)
    if h == EntityType.MEDICATION and t == EntityType.FREQUENCY:
        return RelationType.MEDICATION_FREQUENCY
    if h == EntityType.FREQUENCY and t == EntityType.MEDICATION:
        return RelationType.MEDICATION_FREQUENCY
    return None


def rule_symptom_duration(head: Entity, tail: Entity) -> Optional[RelationType]:
    """SYMPTOM + DURATION → SYMPTOM-DURATION"""
    h, t = _types(head, tail)
    if h == EntityType.SYMPTOM and t == EntityType.DURATION:
        return RelationType.SYMPTOM_DURATION
    if h == EntityType.DURATION and t == EntityType.SYMPTOM:
        return RelationType.SYMPTOM_DURATION
    return None


def rule_symptom_severity(head: Entity, tail: Entity) -> Optional[RelationType]:
    """SYMPTOM + SEVERITY → SYMPTOM-SEVERITY"""
    h, t = _types(head, tail)
    if h == EntityType.SYMPTOM and t == EntityType.SEVERITY:
        return RelationType.SYMPTOM_SEVERITY
    if h == EntityType.SEVERITY and t == EntityType.SYMPTOM:
        return RelationType.SYMPTOM_SEVERITY
    return None


def rule_symptom_body_part(head: Entity, tail: Entity) -> Optional[RelationType]:
    """SYMPTOM + BODY_PART → SYMPTOM-BODY_PART"""
    h, t = _types(head, tail)
    if h == EntityType.SYMPTOM and t == EntityType.BODY_PART:
        return RelationType.SYMPTOM_BODY_PART
    if h == EntityType.BODY_PART and t == EntityType.SYMPTOM:
        return RelationType.SYMPTOM_BODY_PART
    return None


def rule_disease_status(head: Entity, tail: Entity) -> Optional[RelationType]:
    """DISEASE + TEST_RESULT → DISEASE-STATUS (e.g. 'diabetes -- positive')"""
    h, t = _types(head, tail)
    if h == EntityType.DISEASE and t == EntityType.TEST_RESULT:
        return RelationType.DISEASE_STATUS
    if h == EntityType.TEST_RESULT and t == EntityType.DISEASE:
        return RelationType.DISEASE_STATUS
    return None


def rule_test_result(head: Entity, tail: Entity) -> Optional[RelationType]:
    """TEST + TEST_RESULT → TEST-RESULT"""
    h, t = _types(head, tail)
    if h == EntityType.TEST and t == EntityType.TEST_RESULT:
        return RelationType.TEST_RESULT
    if h == EntityType.TEST_RESULT and t == EntityType.TEST:
        return RelationType.TEST_RESULT
    return None


def rule_procedure_date(head: Entity, tail: Entity) -> Optional[RelationType]:
    """PROCEDURE + DATE → PROCEDURE-DATE"""
    h, t = _types(head, tail)
    if h == EntityType.PROCEDURE and t == EntityType.DATE:
        return RelationType.PROCEDURE_DATE
    if h == EntityType.DATE and t == EntityType.PROCEDURE:
        return RelationType.PROCEDURE_DATE
    return None


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ALL_RULES = [
    rule_medication_dosage,
    rule_medication_frequency,
    rule_symptom_duration,
    rule_symptom_severity,
    rule_symptom_body_part,
    rule_disease_status,
    rule_test_result,
    rule_procedure_date,
]
