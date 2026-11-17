"""
Unit tests for Stage 9: Physician Review and Audit Trail.
"""

import pytest
from src.common.schemas import (
    PatientRecord,
    ReviewActionType,
    SectionItem,
    SectionType,
    Span,
)
from src.review.review_engine import PhysicianReviewEngine, DISCLAIMER_NOTICE


@pytest.fixture
def review_setup():
    span = Span(text="aspirin", start=0, end=7, source_sentence_id=0)
    sec_item = SectionItem(
        entity_id="ent-123",
        section=SectionType.DRUG_HISTORY,
        provenance_span=span,
        display_text="aspirin 75 mg",
    )
    record = PatientRecord(
        raw_text="aspirin 75 mg daily",
        section_items=[sec_item],
    )
    engine = PhysicianReviewEngine(patient_record=record)
    return engine, sec_item, record


def test_accept_action(review_setup):
    engine, item, _ = review_setup
    action = engine.accept(item_id=item.id, reviewer="Dr. Alice", note="Confirmed accurate")

    assert action.action == ReviewActionType.ACCEPT
    assert action.reviewer == "Dr. Alice"
    assert engine.get_item_status(item.id) == ReviewActionType.ACCEPT
    assert len(engine.get_audit_log()) == 1


def test_edit_action(review_setup):
    engine, item, record = review_setup
    action = engine.edit(
        item_id=item.id,
        new_value="aspirin 81 mg enteric-coated",
        reviewer="Dr. Bob",
        old_value="aspirin 75 mg",
    )

    assert action.action == ReviewActionType.EDIT
    assert action.new_value == "aspirin 81 mg enteric-coated"
    assert engine.get_item_status(item.id) == ReviewActionType.EDIT
    # Ensure patient record section item display was updated
    assert item.display_text == "aspirin 81 mg enteric-coated"


def test_reject_action(review_setup):
    engine, item, _ = review_setup
    action = engine.reject(item_id=item.id, reviewer="Dr. Charlie", reason="Discontinued last year")

    assert action.action == ReviewActionType.REJECT
    assert engine.get_item_status(item.id) == ReviewActionType.REJECT
    assert action.note == "Discontinued last year"


def test_verify_action_and_disclaimer(review_setup):
    engine, item, _ = review_setup
    action = engine.verify(item_id=item.id, reviewer="Dr. Dana")

    assert action.action == ReviewActionType.VERIFY
    assert engine.get_item_status(item.id) == ReviewActionType.VERIFY
    assert DISCLAIMER_NOTICE in engine.disclaimer
    assert action.DISCLAIMER == DISCLAIMER_NOTICE
