"""
tests/test_ner.py — Pytest test suite for the ClinicalNER engine
================================================================

All tests are self-contained and use only synthetic in-memory data.
No file I/O or network calls are made.

Each test builds a minimal ``PreprocessingOutput`` with a single sentence
whose ``raw_start`` / ``raw_end`` span the full raw text, then calls
``ClinicalNER().run()`` and asserts on the returned ``Entity`` list.
"""

from __future__ import annotations

from typing import List

import pytest

from src.common.schemas import (
    Entity,
    EntityType,
    PreprocessingOutput,
    SentenceInfo,
    Span,
)
from src.ner import ClinicalNER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_output(text: str) -> PreprocessingOutput:
    """
    Build a minimal ``PreprocessingOutput`` with a single sentence.

    The sentence text is assumed to be identical to ``text`` (no
    pre-processing transforms), so ``raw_start = 0`` and
    ``raw_end = len(text)``.
    """
    sent = SentenceInfo(
        sentence_id=0,
        text=text,
        raw_start=0,
        raw_end=len(text),
        tokens=text.split(),
    )
    return PreprocessingOutput(
        raw_text=text,
        normalized_text=text,
        sentences=[sent],
    )


def _types(entities: List[Entity]) -> List[EntityType]:
    return [e.type for e in entities]


def _entity_of(entities: List[Entity], etype: EntityType) -> List[Entity]:
    return [e for e in entities if e.type == etype]


def _has_entity(
    entities: List[Entity],
    etype: EntityType,
    text_fragment: str,
) -> bool:
    """Return True if any entity of ``etype`` contains ``text_fragment``."""
    return any(
        text_fragment.lower() in e.span.text.lower()
        for e in entities
        if e.type == etype
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMedicationDosageFrequency:
    """Paracetamol 500 mg twice a day → MEDICATION + DOSAGE + FREQUENCY."""

    RAW = "Paracetamol 500 mg twice a day"

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_medication_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.MEDICATION, "Paracetamol"), (
            "Expected MEDICATION entity containing 'Paracetamol'"
        )

    def test_dosage_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.DOSAGE, "500 mg"), (
            "Expected DOSAGE entity containing '500 mg'"
        )

    def test_frequency_detected(self) -> None:
        freq_entities = _entity_of(self.entities, EntityType.FREQUENCY)
        assert freq_entities, "Expected at least one FREQUENCY entity"
        matched = any("twice" in e.span.text.lower() for e in freq_entities)
        assert matched, "Expected FREQUENCY entity containing 'twice'"


class TestSymptomDetection:
    """I have fever and headache → two SYMPTOM entities."""

    RAW = "I have fever and headache"

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_fever_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.SYMPTOM, "fever"), (
            "Expected SYMPTOM entity 'fever'"
        )

    def test_headache_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.SYMPTOM, "headache"), (
            "Expected SYMPTOM entity 'headache'"
        )

    def test_at_least_two_symptoms(self) -> None:
        syms = _entity_of(self.entities, EntityType.SYMPTOM)
        assert len(syms) >= 2, f"Expected ≥2 SYMPTOM entities, got {len(syms)}"


class TestDiseaseDetection:
    """known case of hypertension → at least one DISEASE entity."""

    RAW = "known case of hypertension"

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_hypertension_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.DISEASE, "hypertension"), (
            "Expected DISEASE entity 'hypertension'"
        )

    def test_entity_type_is_disease(self) -> None:
        diseases = _entity_of(self.entities, EntityType.DISEASE)
        assert diseases, "Should have at least one DISEASE"


class TestAllergenDetection:
    """allergic to penicillin → ALLERGEN entity."""

    RAW = "allergic to penicillin"

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_allergen_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.ALLERGEN, "penicillin"), (
            "Expected ALLERGEN entity 'penicillin'"
        )


class TestFamilyMemberDetection:
    """My father has diabetes → FAMILY_MEMBER(father) + DISEASE(diabetes)."""

    RAW = "My father has diabetes mellitus"

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_family_member_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.FAMILY_MEMBER, "father"), (
            "Expected FAMILY_MEMBER entity 'father'"
        )

    def test_disease_detected(self) -> None:
        assert _has_entity(self.entities, EntityType.DISEASE, "diabetes mellitus"), (
            "Expected DISEASE entity 'diabetes mellitus'"
        )


class TestOffsetCorrectness:
    """Paracetamol 500 mg — span.start must equal index of 'P' in raw text."""

    RAW = "Paracetamol 500 mg"

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_medication_start_offset(self) -> None:
        meds = [e for e in self.entities if e.type == EntityType.MEDICATION]
        assert meds, "Expected at least one MEDICATION entity"
        paracetamol = next(
            (e for e in meds if "paracetamol" in e.span.text.lower()), None
        )
        assert paracetamol is not None, "Expected 'Paracetamol' MEDICATION entity"
        expected_start = self.RAW.lower().index("paracetamol")
        assert paracetamol.span.start == expected_start, (
            f"Expected span.start={expected_start}, got {paracetamol.span.start}"
        )

    def test_dosage_start_offset(self) -> None:
        dosages = [e for e in self.entities if e.type == EntityType.DOSAGE]
        assert dosages, "Expected at least one DOSAGE entity"
        d = dosages[0]
        expected_start = self.RAW.index("500")
        assert d.span.start == expected_start, (
            f"Expected dosage span.start={expected_start}, got {d.span.start}"
        )

    def test_span_end_is_exclusive(self) -> None:
        """span.end should equal span.start + len(span.text)."""
        for e in self.entities:
            assert e.span.end == e.span.start + len(e.span.text), (
                f"Span end mismatch for entity {e.type}: "
                f"start={e.span.start}, end={e.span.end}, text='{e.span.text}'"
            )


class TestNoDuplicateEntities:
    """No two entities should share the exact same (start, end, type)."""

    RAW = (
        "Patient has hypertension and diabetes mellitus. "
        "She takes metformin 500 mg twice daily."
    )

    def setup_method(self) -> None:
        # Two sentences with different raw_start offsets
        sent0_text = "Patient has hypertension and diabetes mellitus."
        sent1_text = "She takes metformin 500 mg twice daily."
        sents = [
            SentenceInfo(
                sentence_id=0,
                text=sent0_text,
                raw_start=0,
                raw_end=len(sent0_text),
                tokens=sent0_text.split(),
            ),
            SentenceInfo(
                sentence_id=1,
                text=sent1_text,
                raw_start=len(sent0_text) + 1,  # +1 for the space
                raw_end=len(sent0_text) + 1 + len(sent1_text),
                tokens=sent1_text.split(),
            ),
        ]
        output = PreprocessingOutput(
            raw_text=self.RAW,
            normalized_text=self.RAW,
            sentences=sents,
        )
        self.entities = ClinicalNER().run(output)

    def test_no_duplicate_spans(self) -> None:
        seen = set()
        for e in self.entities:
            key = (e.span.start, e.span.end, e.type)
            assert key not in seen, (
                f"Duplicate entity found: type={e.type}, "
                f"start={e.span.start}, end={e.span.end}, text='{e.span.text}'"
            )
            seen.add(key)


class TestEntitySchemaValid:
    """All returned objects must be valid Pydantic Entity models."""

    RAW = "He has fever, cough, and takes amoxicillin 500 mg three times a day."

    def setup_method(self) -> None:
        output = _make_output(self.RAW)
        self.entities = ClinicalNER().run(output)

    def test_all_are_entity_instances(self) -> None:
        assert self.entities, "Expected at least one entity"
        for e in self.entities:
            assert isinstance(e, Entity), (
                f"Expected Entity instance, got {type(e)}"
            )

    def test_all_spans_are_span_instances(self) -> None:
        for e in self.entities:
            assert isinstance(e.span, Span), (
                f"Expected Span instance, got {type(e.span)}"
            )

    def test_confidence_in_range(self) -> None:
        for e in self.entities:
            assert 0.0 <= e.confidence <= 1.0, (
                f"Confidence out of range: {e.confidence}"
            )

    def test_span_start_le_end(self) -> None:
        for e in self.entities:
            assert e.span.start <= e.span.end, (
                f"span.start ({e.span.start}) > span.end ({e.span.end})"
            )

    def test_span_text_nonempty(self) -> None:
        for e in self.entities:
            assert e.span.text, "span.text must not be empty"

    def test_type_is_entity_type_enum(self) -> None:
        for e in self.entities:
            assert isinstance(e.type, EntityType), (
                f"Expected EntityType enum, got {type(e.type)}"
            )

    def test_entity_has_uuid_id(self) -> None:
        for e in self.entities:
            assert e.id, "Entity.id must not be empty"
            # Basic UUID format check (8-4-4-4-12)
            parts = e.id.split("-")
            assert len(parts) == 5, f"Entity.id does not look like a UUID: {e.id}"
