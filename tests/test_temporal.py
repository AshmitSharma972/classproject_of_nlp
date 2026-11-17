"""
Unit tests for Stage 4: Temporal Processing and Timeline Construction.
"""

import pytest
from src.common.schemas import (
    Entity,
    EntityType,
    Span,
    Assertion,
    AssertionStatus,
    Experiencer,
    PreprocessingOutput,
    SentenceInfo,
    TimeExprType,
)
from src.temporal.temporal_engine import TemporalEngine


@pytest.fixture
def temporal_engine():
    return TemporalEngine()


def test_duration_extraction_and_normalization(temporal_engine):
    raw_text = "I have had fever for five days."
    sent = SentenceInfo(
        sentence_id=0,
        text="i have had fever for five days.",
        raw_start=0,
        raw_end=len(raw_text),
        tokens=["i", "have", "had", "fever", "for", "five", "days"],
    )
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="i have had fever for five days.",
        sentences=[sent],
    )
    fever_ent = Entity(
        span=Span(text="fever", start=15, end=20, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )

    time_exprs, timeline = temporal_engine.run(
        entities=[fever_ent], assertions=[], preprocessing_output=prep_out
    )

    assert len(time_exprs) >= 1
    duration_expr = next((t for t in time_exprs if t.type == TimeExprType.DURATION), None)
    assert duration_expr is not None
    assert "5 days" in duration_expr.normalized_value or "five days" in duration_expr.raw_value.lower()
    # Check that fever was linked
    assert fever_ent.id in duration_expr.linked_entity_ids
    # Check provenance
    assert duration_expr.span.start >= 0
    assert duration_expr.span.end <= len(raw_text)
    assert len(timeline) >= 1


def test_relative_dates_and_timeline_order(temporal_engine):
    raw_text = "Yesterday I noticed chills. Today I feel better."
    sent1 = SentenceInfo(
        sentence_id=0,
        text="yesterday i noticed chills.",
        raw_start=0,
        raw_end=27,
        tokens=["yesterday", "i", "noticed", "chills"],
    )
    sent2 = SentenceInfo(
        sentence_id=1,
        text="today i feel better.",
        raw_start=28,
        raw_end=len(raw_text),
        tokens=["today", "i", "feel", "better"],
    )
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="yesterday i noticed chills. today i feel better.",
        sentences=[sent1, sent2],
    )
    chills_ent = Entity(
        span=Span(text="chills", start=20, end=26, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )

    time_exprs, timeline = temporal_engine.run(
        entities=[chills_ent], assertions=[], preprocessing_output=prep_out
    )

    types = [t.type for t in time_exprs]
    assert TimeExprType.RELATIVE in types
    assert len(timeline) >= 1


def test_previous_episode_and_anaphora(temporal_engine):
    raw_text = "I had fever last month. The fever is back now."
    sent1 = SentenceInfo(
        sentence_id=0,
        text="i had fever last month.",
        raw_start=0,
        raw_end=23,
        tokens=["i", "had", "fever", "last", "month"],
    )
    sent2 = SentenceInfo(
        sentence_id=1,
        text="the fever is back now.",
        raw_start=24,
        raw_end=len(raw_text),
        tokens=["the", "fever", "is", "back", "now"],
    )
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="i had fever last month. the fever is back now.",
        sentences=[sent1, sent2],
    )
    fever1 = Entity(
        span=Span(text="fever", start=6, end=11, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )

    time_exprs, timeline = temporal_engine.run(
        entities=[fever1], assertions=[], preprocessing_output=prep_out
    )

    assert len(time_exprs) >= 1
    # Check provenance
    for te in time_exprs:
        assert te.span.start >= 0
        assert te.span.end <= len(raw_text)
