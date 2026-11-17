"""
Unit tests for Stage 6: Section Classification.
"""

import pytest
from src.common.schemas import (
    Entity,
    EntityType,
    Span,
    Assertion,
    AssertionStatus,
    Experiencer,
    SectionType,
)
from src.section_classification.section_classifier import SectionClassifier


@pytest.fixture
def classifier():
    return SectionClassifier()


def test_chief_complaint_and_hpi_distribution(classifier):
    fever = Entity(
        span=Span(text="fever", start=0, end=5, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )
    cough = Entity(
        span=Span(text="cough", start=10, end=15, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )
    fever_assertion = Assertion(
        entity_id=fever.id, status=AssertionStatus.PRESENT, experiencer=Experiencer.PATIENT
    )
    cough_assertion = Assertion(
        entity_id=cough.id, status=AssertionStatus.PRESENT, experiencer=Experiencer.PATIENT
    )

    items = classifier.run(
        entities=[fever, cough],
        assertions=[fever_assertion, cough_assertion],
        time_expressions=[],
        relations=[],
    )

    sections = {item.entity_id: item.section for item in items}
    # First symptom becomes Chief Complaint, subsequent symptoms go to HPI
    assert sections[fever.id] == SectionType.CHIEF_COMPLAINT
    assert sections[cough.id] == SectionType.HPI


def test_family_history_separation(classifier):
    dm = Entity(
        span=Span(text="diabetes", start=15, end=23, source_sentence_id=0),
        type=EntityType.DISEASE,
        confidence=1.0,
    )
    dm_assertion = Assertion(
        entity_id=dm.id,
        status=AssertionStatus.FAMILY,
        experiencer=Experiencer.OTHER,
        trigger_text="father",
    )

    items = classifier.run(
        entities=[dm],
        assertions=[dm_assertion],
        time_expressions=[],
        relations=[],
    )

    assert len(items) == 1
    # Family history facts must never become patient personal/past history
    assert items[0].section == SectionType.FAMILY_HISTORY
    assert items[0].section != SectionType.PAST_HISTORY
    assert items[0].section != SectionType.PERSONAL_HISTORY


def test_negated_symptoms_in_ros(classifier):
    vomiting = Entity(
        span=Span(text="vomiting", start=5, end=13, source_sentence_id=0),
        type=EntityType.SYMPTOM,
        confidence=1.0,
    )
    vomiting_assertion = Assertion(
        entity_id=vomiting.id,
        status=AssertionStatus.NEGATED,
        experiencer=Experiencer.PATIENT,
        trigger_text="no",
    )

    items = classifier.run(
        entities=[vomiting],
        assertions=[vomiting_assertion],
        time_expressions=[],
        relations=[],
    )

    assert len(items) == 1
    assert items[0].section == SectionType.ROS
    assert "No vomiting (negated)" in items[0].display_text


def test_allergen_and_medication_sections(classifier):
    penicillin = Entity(
        span=Span(text="penicillin", start=0, end=10, source_sentence_id=0),
        type=EntityType.ALLERGEN,
        confidence=1.0,
    )
    aspirin = Entity(
        span=Span(text="aspirin", start=15, end=22, source_sentence_id=0),
        type=EntityType.MEDICATION,
        confidence=1.0,
    )
    pen_ass = Assertion(entity_id=penicillin.id, status=AssertionStatus.PRESENT)
    asp_ass = Assertion(entity_id=aspirin.id, status=AssertionStatus.PRESENT)

    items = classifier.run(
        entities=[penicillin, aspirin],
        assertions=[pen_ass, asp_ass],
        time_expressions=[],
        relations=[],
    )

    sections = {item.entity_id: item.section for item in items}
    assert sections[penicillin.id] == SectionType.ALLERGY
    assert sections[aspirin.id] == SectionType.DRUG_HISTORY
