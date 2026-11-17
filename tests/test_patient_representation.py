"""
Unit tests for Stage 7: Patient Representation.
"""

import json
import pytest
from src.common.schemas import (
    Entity,
    EntityType,
    Span,
    Assertion,
    AssertionStatus,
    Experiencer,
    SectionItem,
    SectionType,
    PreprocessingOutput,
    SentenceInfo,
    TimelineEntry,
)
from src.patient_representation.patient_builder import PatientRecordBuilder


@pytest.fixture
def builder():
    return PatientRecordBuilder()


def test_patient_record_assembly_and_provenance(builder):
    raw_text = "I have had fever for 5 days. I do not have diabetes."
    sent1 = SentenceInfo(sentence_id=0, text="i have had fever for 5 days.", raw_start=0, raw_end=28, tokens=[])
    sent2 = SentenceInfo(sentence_id=1, text="i do not have diabetes.", raw_start=29, raw_end=len(raw_text), tokens=[])
    prep_out = PreprocessingOutput(
        raw_text=raw_text,
        normalized_text="i have had fever for 5 days. i do not have diabetes.",
        sentences=[sent1, sent2],
    )

    fever = Entity(
        span=Span(text="fever", start=11, end=16, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )
    dm = Entity(
        span=Span(text="diabetes", start=43, end=51, source_sentence_id=1),
        type=EntityType.DISEASE,
        confidence=1.0,
    )
    fever_ass = Assertion(entity_id=fever.id, status=AssertionStatus.PRESENT)
    dm_ass = Assertion(entity_id=dm.id, status=AssertionStatus.NEGATED, trigger_text="not")

    sec_fever = SectionItem(
        entity_id=fever.id,
        section=SectionType.CHIEF_COMPLAINT,
        provenance_span=fever.span,
        display_text="fever",
    )
    sec_dm = SectionItem(
        entity_id=dm.id,
        section=SectionType.ROS,
        provenance_span=dm.span,
        display_text="No diabetes (negated)",
    )
    timeline_entry = TimelineEntry(order=0, label="fever — 5 days", raw_text="5 days")

    record = builder.build(
        raw_text=raw_text,
        preprocessing_output=prep_out,
        entities=[fever, dm],
        assertions=[fever_ass, dm_ass],
        time_expressions=[],
        relations=[],
        section_items=[sec_fever, sec_dm],
        timeline=[timeline_entry],
    )

    # Verify provenance and schema
    assert record.raw_text == raw_text
    assert len(record.entities) == 2
    assert record.entities[0].span.start == 11
    assert record.entities[0].span.end == 16
    assert record.entities[0].span.source_sentence_id == 0

    # Verify sections grouped
    assert "Chief Complaint" in record.sections
    assert "ROS" in record.sections
    assert "fever" in record.sections["Chief Complaint"]

    # Verify helper methods
    retrieved = record.get_entity_by_id(fever.id)
    assert retrieved is not None
    assert retrieved.span.text == "fever"

    ass_retrieved = record.get_assertion_for_entity(dm.id)
    assert ass_retrieved is not None
    assert ass_retrieved.status == AssertionStatus.NEGATED

    # Verify JSON compatibility
    json_str = record.model_dump_json()
    assert isinstance(json_str, str)
    data = json.loads(json_str)
    assert data["id"] == record.id
    assert len(data["entities"]) == 2
