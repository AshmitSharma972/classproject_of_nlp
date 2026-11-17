"""
Section Classification Rules for the clinical NLP pipeline.

These deterministic rules assign each entity (with its assertion and
temporal context) to one of the 8 standard clinical note sections.

Rule priority (highest first):
  1. FAMILY_HISTORY  -- experiencer=OTHER
  2. ALLERGY         -- entity type is ALLERGEN
  3. DRUG_HISTORY    -- entity type is MEDICATION, status is PRESENT
  4. PAST_HISTORY    -- status is HISTORICAL, or entity has past-temporal anchor
  5. CHIEF_COMPLAINT -- first PRESENT SYMPTOM(s) for the patient
  6. HPI             -- remaining PRESENT SYMPTOMs and related entities
  7. ROS             -- negated symptoms (NEGATED status)
  8. PERSONAL_HISTORY -- DISEASE with PRESENT status (ongoing conditions)

Notes:
- Negated entities retain their status; negation is never stripped.
- FAMILY entities are never reclassified as patient history.
- Section classification operates on the combined (entity + assertion + temporal) tuple.
"""

from __future__ import annotations

from typing import List, Optional, Set

from src.common.schemas import (
    Assertion, AssertionStatus, Entity, EntityType,
    Experiencer, SectionType, TimeExpr, TimeExprType,
)


def classify_entity(
    entity: Entity,
    assertion: Optional[Assertion],
    time_expressions: List[TimeExpr],
    chief_complaint_filled: bool,
) -> SectionType:
    """
    Classify a single entity into a clinical section.

    Args:
        entity: The entity to classify.
        assertion: The assertion for this entity (or None → defaults to PRESENT/PATIENT).
        time_expressions: All time expressions linked to this entity.
        chief_complaint_filled: True if at least one Chief Complaint has been assigned.

    Returns:
        The appropriate SectionType.
    """
    status = assertion.status if assertion else AssertionStatus.PRESENT
    experiencer = assertion.experiencer if assertion else Experiencer.PATIENT

    # ------------------------------------------------------------------ #
    # Rule 1: FAMILY HISTORY -- experiencer is OTHER (family member)
    # ------------------------------------------------------------------ #
    if experiencer == Experiencer.OTHER:
        return SectionType.FAMILY_HISTORY

    # ------------------------------------------------------------------ #
    # Rule 2: ALLERGY -- entity type is ALLERGEN
    # ------------------------------------------------------------------ #
    if entity.type == EntityType.ALLERGEN:
        return SectionType.ALLERGY

    # Keep every absent finding out of present-illness sections, including
    # negated diseases and measurements.
    if status == AssertionStatus.NEGATED:
        return SectionType.ROS

    # ------------------------------------------------------------------ #
    # Rule 3: DRUG HISTORY -- medication currently being taken
    # ------------------------------------------------------------------ #
    if entity.type == EntityType.MEDICATION and status in (
        AssertionStatus.PRESENT, AssertionStatus.HISTORICAL
    ):
        if status == AssertionStatus.HISTORICAL:
            return SectionType.PAST_HISTORY
        return SectionType.DRUG_HISTORY

    # ------------------------------------------------------------------ #
    # Rule 4: PAST HISTORY -- HISTORICAL status
    # ------------------------------------------------------------------ #
    if status == AssertionStatus.HISTORICAL:
        return SectionType.PAST_HISTORY

    # Also classify as Past History if only past temporal anchors are linked
    if time_expressions:
        all_past = all(
            te.type in (TimeExprType.DATE, TimeExprType.PREVIOUS_EPISODE)
            for te in time_expressions
        )
        has_past_anchor = any(
            te.type in (TimeExprType.DATE, TimeExprType.PREVIOUS_EPISODE)
            for te in time_expressions
        )
        if all_past and has_past_anchor and status != AssertionStatus.PRESENT:
            return SectionType.PAST_HISTORY

    # ------------------------------------------------------------------ #
    # Rule 5 & 6: Chief Complaint vs HPI -- PRESENT SYMPTOMs
    # ------------------------------------------------------------------ #
    if entity.type == EntityType.SYMPTOM and status == AssertionStatus.PRESENT:
        if not chief_complaint_filled:
            return SectionType.CHIEF_COMPLAINT
        return SectionType.HPI

    # ------------------------------------------------------------------ #
    # Rule 7: ROS -- negated symptoms
    # ------------------------------------------------------------------ #
    if entity.type == EntityType.SYMPTOM and status == AssertionStatus.NEGATED:
        return SectionType.ROS

    # ------------------------------------------------------------------ #
    # Rule 8: PERSONAL HISTORY -- ongoing diseases (PRESENT)
    # ------------------------------------------------------------------ #
    if entity.type == EntityType.DISEASE and status == AssertionStatus.PRESENT:
        return SectionType.PERSONAL_HISTORY

    # ------------------------------------------------------------------ #
    # Fallbacks
    # ------------------------------------------------------------------ #
    if entity.type in (EntityType.TEST, EntityType.TEST_RESULT):
        return SectionType.HPI

    if entity.type == EntityType.PROCEDURE:
        if status == AssertionStatus.HISTORICAL:
            return SectionType.PAST_HISTORY
        return SectionType.HPI

    if entity.type in (EntityType.DOSAGE, EntityType.FREQUENCY):
        return SectionType.DRUG_HISTORY

    if entity.type == EntityType.BODY_PART:
        return SectionType.HPI

    if entity.type in (EntityType.DURATION, EntityType.DATE):
        return SectionType.HPI

    if entity.type == EntityType.SEVERITY:
        return SectionType.HPI

    if entity.type == EntityType.FAMILY_MEMBER:
        return SectionType.FAMILY_HISTORY

    # Default fallback
    return SectionType.HPI
