"""
Unit tests for Stage 5: Relation Extraction.
"""

import pytest
from src.common.schemas import (
    Entity,
    EntityType,
    Span,
    RelationType,
    PreprocessingOutput,
    SentenceInfo,
)
from src.relation_extraction.relation_engine import RelationEngine


@pytest.fixture
def relation_engine():
    return RelationEngine()


def test_medication_dosage_frequency_relations(relation_engine):
    raw_text = "Paracetamol 500 mg twice a day"
    sent = SentenceInfo(
        sentence_id=0,
        text="paracetamol 500 mg twice a day",
        raw_start=0,
        raw_end=len(raw_text),
        tokens=["paracetamol", "500", "mg", "twice", "a", "day"],
    )
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="paracetamol 500 mg twice a day",
        sentences=[sent],
    )

    med = Entity(
        span=Span(text="Paracetamol", start=0, end=11, source_sentence_id=0),
        type=EntityType.MEDICATION,
        confidence=1.0,
    )
    dosage = Entity(
        span=Span(text="500 mg", start=12, end=18, source_sentence_id=0),
        type=EntityType.DOSAGE,
        confidence=0.85,
    )
    freq = Entity(
        span=Span(text="twice a day", start=19, end=30, source_sentence_id=0),
        type=EntityType.FREQUENCY,
        confidence=0.85,
    )

    relations = relation_engine.run(
        entities=[med, dosage, freq],
        assertions=[],
        time_expressions=[],
        preprocessing_output=prep_out,
    )

    rel_types = [r.relation_type for r in relations]
    assert RelationType.MEDICATION_DOSAGE in rel_types
    assert RelationType.MEDICATION_FREQUENCY in rel_types

    # Verify source sentence ID is preserved
    for r in relations:
        assert r.source_sentence_id == 0
        assert r.evidence_text is not None


def test_symptom_duration_severity_body_part(relation_engine):
    raw_text = "Severe chest pain for 3 days"
    sent = SentenceInfo(
        sentence_id=0,
        text="severe chest pain for 3 days",
        raw_start=0,
        raw_end=len(raw_text),
        tokens=["severe", "chest", "pain", "for", "3", "days"],
    )
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="severe chest pain for 3 days",
        sentences=[sent],
    )

    sev = Entity(
        span=Span(text="Severe", start=0, end=6, source_sentence_id=0),
        type=EntityType.SEVERITY,
        confidence=1.0,
    )
    bp = Entity(
        span=Span(text="chest", start=7, end=12, source_sentence_id=0),
        type=EntityType.BODY_PART,
        confidence=1.0,
    )
    sym = Entity(
        span=Span(text="pain", start=13, end=17, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )
    dur = Entity(
        span=Span(text="3 days", start=22, end=28, source_sentence_id=0),
        type=EntityType.DURATION,
        confidence=0.85,
    )

    relations = relation_engine.run(
        entities=[sev, bp, sym, dur],
        assertions=[],
        time_expressions=[],
        preprocessing_output=prep_out,
    )

    rel_types = [r.relation_type for r in relations]
    assert RelationType.SYMPTOM_SEVERITY in rel_types
    assert RelationType.SYMPTOM_BODY_PART in rel_types
    assert RelationType.SYMPTOM_DURATION in rel_types


def test_cross_sentence_candidate_exclusion(relation_engine):
    raw_text = "Patient takes Metformin. The dosage is 500 mg."
    sent1 = SentenceInfo(
        sentence_id=0,
        text="patient takes metformin.",
        raw_start=0,
        raw_end=24,
        tokens=["patient", "takes", "metformin"],
    )
    sent2 = SentenceInfo(
        sentence_id=1,
        text="the dosage is 500 mg.",
        raw_start=25,
        raw_end=len(raw_text),
        tokens=["the", "dosage", "is", "500", "mg"],
    )
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="patient takes metformin. the dosage is 500 mg.",
        sentences=[sent1, sent2],
    )

    med = Entity(
        span=Span(text="Metformin", start=14, end=23, source_sentence_id=0),
        type=EntityType.MEDICATION,
        confidence=1.0,
    )
    dosage = Entity(
        span=Span(text="500 mg", start=39, end=45, source_sentence_id=1),
        type=EntityType.DOSAGE,
        confidence=0.85,
    )

    # Entities are in different sentences, should not form relations in sentence-level extractor
    relations = relation_engine.run(
        entities=[med, dosage],
        assertions=[],
        time_expressions=[],
        preprocessing_output=prep_out,
    )

    assert len(relations) == 0
