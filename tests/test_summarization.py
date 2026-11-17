"""
Unit tests for Stage 8: Summarization.
"""

import pytest
from src.common.schemas import (
    PatientRecord,
    TimelineEntry,
)
from src.summarization.summarizer import ClinicalSummarizer


@pytest.fixture
def summarizer():
    return ClinicalSummarizer()


def test_summary_generated_from_structured_record_only(summarizer):
    record = PatientRecord(
        raw_text="The patient says something raw here that should not directly bypass structured extraction.",
        sections={
            "Chief Complaint": ["fever (5 days)"],
            "HPI": ["cough"],
            "Past History": ["tuberculosis (historical)"],
            "Drug History": ["Metformin 500 mg"],
            "Allergy": ["penicillin"],
            "Family History": ["diabetes [father]"],
            "ROS": ["No chest pain (negated)"],
        },
        timeline=[TimelineEntry(order=0, label="fever — 5 days")],
    )

    summary = summarizer.summarize(record)

    # Must contain sections derived from structured fields
    assert "CHIEF COMPLAINT" in summary
    assert "fever (5 days)" in summary
    assert "CURRENT MEDICATIONS" in summary
    assert "Metformin 500 mg" in summary
    assert "FAMILY HISTORY" in summary
    assert "diabetes [father]" in summary
    assert "REVIEW OF SYSTEMS (NEGATED / ABSENT)" in summary
    assert "No chest pain (negated)" in summary
    assert "CLINICAL TIMELINE" in summary
    assert "Step 1: fever — 5 days" in summary

    # Invariants: Negated facts remain negated, family remains family
    assert "chest pain" not in summary.split("CHIEF COMPLAINT")[1].split("HISTORY OF PRESENT ILLNESS")[0]
    assert "No chest pain (negated)" in summary

    # Mandatory decision support notice must be present
    assert "Decision support only. The physician remains responsible for clinical decisions." in summary


def test_summary_never_hallucinates_without_section_data(summarizer):
    empty_record = PatientRecord(
        raw_text="Random conversational narration without extracted entities.",
        sections={},
        timeline=[],
    )
    summary = summarizer.summarize(empty_record)

    # Empty sections should not have spurious clinical findings
    assert "Chief Complaint" not in summary
    assert "Current Medications" not in summary
    assert "Decision support only" in summary
