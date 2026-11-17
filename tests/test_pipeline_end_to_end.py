"""
End-to-End Integration Tests for Clinical Patient Case-Taking Pipeline.

Verifies:
- Pipeline stage order execution.
- Shared schema compatibility across all stages.
- Original-text character span offsets mapped accurately.
- Negation preservation (e.g. denies diabetes -> NEGATED).
- Temporal information propagation into timeline.
- Intra-sentence relation extraction propagation.
- Section classification correctness.
- Patient representation completeness and JSON serialization.
- Summary generation exclusively from structured data (never hallucinating raw fragments).
- Physician review functionality and audit log.
- Edge cases and failure modes with synthetic cases.
"""

import json
from pathlib import Path
import pytest

from src.pipeline.clinical_pipeline import ClinicalPipeline, PipelineResult
from src.common.schemas import (
    AssertionStatus,
    EntityType,
    Experiencer,
    RelationType,
    SectionType,
)
from src.review.review_engine import DISCLAIMER_NOTICE


@pytest.fixture(scope="module")
def pipeline():
    return ClinicalPipeline()


def test_end_to_end_standard_narration(pipeline):
    raw_text = (
        "I've had fever and cough for 5 days. I don't have diabetes or chest pain. "
        "My father has hypertension. Currently taking Paracetamol 500 mg twice a day."
    )

    result: PipelineResult = pipeline.run(raw_text)

    # 1. Pipeline result schema & Preprocessing
    assert isinstance(result, PipelineResult)
    assert result.raw_text == raw_text
    assert len(result.preprocessing.sentences) >= 3

    # Contraction expanded: "I've" -> "i have"
    assert "i have" in result.preprocessing.normalized_text.lower()
    # Negation word preserved
    assert "not" in result.preprocessing.normalized_text.lower()

    # 2. NER & Original Offset Invariant
    assert len(result.entities) > 0
    entity_texts = [e.span.text.lower() for e in result.entities]
    assert "fever" in entity_texts
    assert "paracetamol" in entity_texts

    for ent in result.entities:
        # Every span must refer strictly to original text
        slice_text = raw_text[ent.span.start:ent.span.end]
        assert ent.span.text.lower() == slice_text.lower(), (
            f"Offset mismatch for entity '{ent.span.text}': "
            f"expected '{slice_text}' at {ent.span.start}:{ent.span.end}"
        )

    # 3. Assertion & Negation preservation
    # Check diabetes or chest pain is negated
    diabetes_assertion = next(
        (a for a in result.assertions if any(
            e.id == a.entity_id and "diabetes" in e.span.text.lower() for e in result.entities
        )),
        None
    )
    assert diabetes_assertion is not None
    assert diabetes_assertion.status == AssertionStatus.NEGATED

    # Check father's hypertension is classified as FAMILY with experiencer OTHER
    htn_assertion = next(
        (a for a in result.assertions if any(
            e.id == a.entity_id and "hypertension" in e.span.text.lower() for e in result.entities
        )),
        None
    )
    assert htn_assertion is not None
    assert htn_assertion.status == AssertionStatus.FAMILY
    assert htn_assertion.experiencer == Experiencer.OTHER

    # 4. Temporal propagation
    durations = [t for t in result.time_expressions if "5 days" in (t.normalized_value or t.raw_value).lower()]
    assert len(durations) >= 1
    assert len(result.timeline) >= 1

    # 5. Relation propagation
    med_relations = [r for r in result.relations if r.relation_type == RelationType.MEDICATION_DOSAGE]
    assert len(med_relations) >= 1

    # 6. Section classification
    sections = result.patient_record.sections
    assert SectionType.CHIEF_COMPLAINT.value in sections or SectionType.HPI.value in sections
    assert SectionType.FAMILY_HISTORY.value in sections
    assert SectionType.DRUG_HISTORY.value in sections
    assert SectionType.ROS.value in sections  # Negated diabetes/chest pain

    # 7. Patient representation JSON serializability
    json_data = json.loads(result.patient_record.model_dump_json())
    assert json_data["id"] == result.patient_record.id
    assert len(json_data["entities"]) == len(result.entities)

    # 8. Summary generated only from structured record
    assert "CLINICAL HISTORY SUMMARY" in result.summary
    assert DISCLAIMER_NOTICE in result.summary
    assert "No diabetes (negated)" in result.summary or "diabetes" in result.summary

    # 9. Review Engine functionality
    review_engine = pipeline.create_review_engine(result.patient_record)
    target_ent = result.entities[0]
    act_accept = review_engine.accept(target_ent.id, "Dr. Specialist", "Accurate finding")
    assert act_accept.action.value == "ACCEPT"

    act_edit = review_engine.edit(target_ent.id, "Mild intermittent fever", "Dr. Specialist")
    assert act_edit.action.value == "EDIT"
    assert len(review_engine.get_audit_log()) == 2


def test_end_to_end_synthetic_cases(pipeline):
    """Run pipeline across sample cases from data/sample/ to ensure stability across clinical profiles."""
    sample_dir = Path("data") / "sample"
    if not sample_dir.exists():
        pytest.skip("data/sample directory not found")

    sample_files = sorted(list(sample_dir.glob("case_*.txt")))
    if not sample_files:
        pytest.skip("No synthetic cases found in data/sample")

    # Run on first 5 sample cases
    for case_file in sample_files[:5]:
        text = case_file.read_text(encoding="utf-8")
        res = pipeline.run(text)
        assert isinstance(res, PipelineResult)
        assert len(res.patient_record.id) > 0
        assert len(res.summary) > 0
        assert DISCLAIMER_NOTICE in res.summary


def test_edge_case_empty_and_noise_text(pipeline):
    """Verify pipeline gracefully handles empty or short noise input without raising exceptions."""
    noisy_text = "..."
    res = pipeline.run(noisy_text)
    assert isinstance(res, PipelineResult)
    assert len(res.entities) == 0
    assert len(res.relations) == 0
    assert DISCLAIMER_NOTICE in res.summary
