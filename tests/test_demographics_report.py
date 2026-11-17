from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.schemas import DemographicsInput
from src.pipeline import ClinicalPipeline
from src.review.safety_flags import build_safety_flags
from web_api import _create_docx_report


def valid_details(**overrides):
    value = {"full_name": "Fever Kumar", "age": 34, "sex": "Not stated", "consent": True}
    value.update(overrides)
    return value


@pytest.mark.parametrize("details", [
    valid_details(age=121), valid_details(age=-1), valid_details(phone="123-4567"),
    valid_details(phone="١٢٣٤٥٦٧٨"),
    valid_details(full_name="A"), valid_details(visit_date=date.today() + timedelta(days=1)),
    valid_details(consent=False), valid_details(age=None, date_of_birth=None),
])
def test_demographics_reject_invalid_required_or_consent_values(details):
    with pytest.raises(ValidationError):
        DemographicsInput.model_validate(details)


def test_demographics_accept_dob_and_calculate_age():
    details = DemographicsInput.model_validate(valid_details(age=None, date_of_birth=date(2000, 1, 1)))
    assert details.age is not None and 0 <= details.age <= 120


def test_demographic_name_does_not_enter_clinical_pipeline():
    for name in ("Fever Kumar", "Diabetes Rao"):
        demographics = DemographicsInput.model_validate(valid_details(full_name=name))
        assert demographics.full_name == name
        result = ClinicalPipeline().run("Patient reports a cough.")
        texts = {entity.span.text.casefold() for entity in result.entities}
        assert "fever" not in texts and "diabetes" not in texts


def test_narration_demographic_conflicts_are_flags():
    flags = build_safety_flags("I am 45 years old and allergic to penicillin.", [], [], [], [],
                               demographics={"age": 34, "known_allergies": "none"})
    messages = [flag["message"] for flag in flags]
    assert any("Narration says 45" in text and "form says 34" in text for text in messages)
    assert any("allergy" in text.lower() and "please verify" in text.lower() for text in messages)


def test_allergy_conflict_detects_different_named_allergen_and_respects_negation():
    different = build_safety_flags("Patient is allergic to aspirin.", [], [], [], [],
                                   demographics={"age": None, "known_allergies": "penicillin"})
    assert any(flag["id"] == "demographic-allergy-conflict" for flag in different)
    negated = build_safety_flags("Patient is not allergic to penicillin.", [], [], [], [],
                                 demographics={"age": None, "known_allergies": "none"})
    assert not any(flag["id"] == "demographic-allergy-conflict" for flag in negated)


def test_default_report_view_is_human_readable_and_json_is_developer_only():
    html = (Path(__file__).parents[1] / "index.html").read_text(encoding="utf-8")
    assert 'data-pane="report"' in html
    assert 'data-tab="report"' in html
    assert "Clinical Information Summary" in html
    assert "Developer view" in html
    assert 'data-pane="json"' not in html
    assert "This prototype is not compliant with medical privacy regulations by itself." not in html
    assert "Decision-support summary only. Not a diagnosis or treatment recommendation. Must be reviewed by a licensed physician." in html
    report_markup = html.split('data-pane="report"', 1)[1].split('data-pane="summary-evidence"', 1)[0]
    for heading in ("Chief complaints", "History of present illness", "Past medical history", "Family history",
                    "Current medications", "Allergies", "Vitals and test results", "Timeline",
                    "Safety and completeness flags", "Pipeline provenance", "Physician review status",
                    "Physician sign-off", "Required disclaimer"):
        assert heading in report_markup
    assert "raw_text" not in report_markup and "patient_record" not in report_markup


def test_docx_report_renders_missing_optional_fields_as_not_provided():
    from io import BytesIO
    import docx
    demographics = DemographicsInput.model_validate(valid_details()).model_dump(exclude={"consent"})
    document_bytes = _create_docx_report({"patient_record": {}, "flags": []}, demographics)
    document = docx.Document(BytesIO(document_bytes))
    text = "\n".join(p.text for p in document.paragraphs)
    text += "\n" + "\n".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    assert "Not provided" in text
    assert "Decision-support summary only" in text
