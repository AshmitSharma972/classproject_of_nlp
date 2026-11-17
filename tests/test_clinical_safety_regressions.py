"""Safety regressions derived from clinician-facing pipeline requirements.

These tests intentionally codify clause-level attribution and provenance
requirements.  They use only synthetic text; no patient data is involved.
"""

from datetime import date

import pytest

from src.common.schemas import AssertionStatus, EntityType
from src.pipeline import ClinicalPipeline
from src.preprocessing import ClinicalPreprocessor
from src.temporal import TemporalEngine


@pytest.fixture(scope="module")
def pipeline():
    return ClinicalPipeline()


def entities_by_text(result):
    return {entity.span.text.lower(): entity for entity in result.entities}


def assertion_for(result, text):
    entity = entities_by_text(result)[text.lower()]
    return next(item for item in result.assertions if item.entity_id == entity.id)


@pytest.mark.parametrize(
    ("text", "entity", "expected"),
    [
        ("no fever but cough for 3 days", "fever", AssertionStatus.NEGATED),
        ("no fever but cough for 3 days", "cough", AssertionStatus.PRESENT),
        ("denies chest pain", "chest pain", AssertionStatus.NEGATED),
        ("no history of diabetes", "diabetes", AssertionStatus.NEGATED),
        ("rule out TB", "tb", AssertionStatus.POSSIBLE),
        ("no change in pain", "pain", AssertionStatus.PRESENT),
        ("cannot deny fever", "fever", AssertionStatus.PRESENT),
        ("not only cough but also fever", "cough", AssertionStatus.PRESENT),
        ("not only cough but also fever", "fever", AssertionStatus.PRESENT),
        ("not certain pneumonia", "pneumonia", AssertionStatus.POSSIBLE),
    ],
)
def test_assertion_clause_and_pseudonegation_safety(pipeline, text, entity, expected):
    assert assertion_for(pipeline.run(text), entity).status == expected


def test_family_subject_does_not_leak_to_patient_clause(pipeline):
    result = pipeline.run("My father had diabetes. I have fever.")
    assert assertion_for(result, "diabetes").status == AssertionStatus.FAMILY
    assert assertion_for(result, "fever").status == AssertionStatus.PRESENT


def test_family_subject_is_clause_local(pipeline):
    result = pipeline.run("Mother has diabetes and I have hypertension")
    assert assertion_for(result, "diabetes").status == AssertionStatus.FAMILY
    assert assertion_for(result, "hypertension").status == AssertionStatus.PRESENT


def test_duration_links_only_to_its_nearest_clause_entity(pipeline):
    result = pipeline.run("fever for 5 days and cough since yesterday")
    links = {
        time_expr.normalized_value: {
            next(entity.span.text.lower() for entity in result.entities if entity.id == entity_id)
            for entity_id in time_expr.linked_entity_ids
        }
        for time_expr in result.time_expressions
    }
    assert links["5 days"] == {"fever"}
    assert links["yesterday"] == {"cough"}


def test_duration_normalizes_week_and_childhood(pipeline):
    result = pipeline.run("cough for 1 week. Asthma since childhood.")
    normalized = {item.normalized_value for item in result.time_expressions}
    assert "7 days" in normalized
    assert "childhood" in normalized


def test_relative_dates_use_supplied_encounter_date_not_clock():
    engine = TemporalEngine(encounter_date=date(2026, 1, 10))
    prep = ClinicalPreprocessor().run("fever since yesterday")
    entities = ClinicalPipeline().ner_engine.run(prep)
    assertions = ClinicalPipeline().assertion_engine.run(entities, prep)
    expressions, _ = engine.run(entities, assertions, prep)
    assert any(item.normalized_value == "2026-01-09" for item in expressions)


def test_medication_doses_do_not_cross_link_and_frequency_is_shared(pipeline):
    result = pipeline.run("Metformin 500 mg and amlodipine 5 mg twice daily")
    labels = {(next(e.span.text.lower() for e in result.entities if e.id == r.head_entity_id),
               r.relation_type.value,
               next(e.span.text.lower() for e in result.entities if e.id == r.tail_entity_id))
              for r in result.relations}
    assert ("metformin", "MEDICATION-DOSAGE", "500 mg") in labels
    assert ("amlodipine", "MEDICATION-DOSAGE", "5 mg") in labels
    assert ("metformin", "MEDICATION-DOSAGE", "5 mg") not in labels
    assert ("amlodipine", "MEDICATION-FREQUENCY", "twice daily") in labels


def test_three_drug_doses_bind_to_nearest_preceding_drug(pipeline):
    result = pipeline.run("Metformin 500 mg, aspirin 75 mg, and amlodipine 5 mg once daily")
    entities = {e.id: e.span.text.lower() for e in result.entities}
    labels = {(entities[r.head_entity_id], r.relation_type.value, entities[r.tail_entity_id]) for r in result.relations}
    assert ("metformin", "MEDICATION-DOSAGE", "500 mg") in labels
    assert ("aspirin", "MEDICATION-DOSAGE", "75 mg") in labels
    assert ("amlodipine", "MEDICATION-DOSAGE", "5 mg") in labels
    assert {pair for pair in labels if pair[1] == "MEDICATION-DOSAGE"} == {
        ("metformin", "MEDICATION-DOSAGE", "500 mg"),
        ("aspirin", "MEDICATION-DOSAGE", "75 mg"),
        ("amlodipine", "MEDICATION-DOSAGE", "5 mg"),
    }


def test_common_abbreviations_and_labs_are_clinically_typed(pipeline):
    result = pipeline.run("HTN, DM, T2DM, SOB, c/o sugar. BP 140/90. Hb 9.2 g/dL.")
    by_text = entities_by_text(result)
    assert by_text["htn"].type == EntityType.DISEASE
    assert by_text["dm"].type == EntityType.DISEASE
    assert by_text["t2dm"].type == EntityType.DISEASE
    assert by_text["htn"].normalized_text == "hypertension"
    assert by_text["dm"].normalized_text == "diabetes mellitus"
    assert by_text["sob"].type == EntityType.SYMPTOM
    assert by_text["sugar"].type == EntityType.TEST_RESULT
    assert by_text["bp"].type == EntityType.TEST
    assert by_text["140/90"].type == EntityType.TEST_RESULT
    assert by_text["hb"].type == EntityType.TEST
    assert by_text["9.2 g/dl"].type == EntityType.TEST_RESULT


def test_vital_measurements_keep_value_unit_and_original_offsets(pipeline):
    text = "Temp 38.2 C, HR 92 bpm, SpO2 97%, glucose 110 mg/dL."
    result = pipeline.run(text)
    measured = {e.span.text: e for e in result.entities if e.type == EntityType.TEST_RESULT}
    for value in ("38.2 C", "92 bpm", "97%", "110 mg/dL"):
        assert value in measured
        entity = measured[value]
        assert text[entity.span.start:entity.span.end] == value


def test_allergen_is_not_a_medication(pipeline):
    result = pipeline.run("allergic to penicillin")
    penicillin = entities_by_text(result)["penicillin"]
    assert penicillin.type == EntityType.ALLERGEN


def test_preprocessing_preserves_abbreviations_decimals_and_raw_offsets():
    raw = "Dr. Rao prescribed 5.5 mg b.i.d. for fever."
    output = ClinicalPreprocessor().run(raw)
    assert len(output.sentences) == 1
    assert output.sentences[0].text == output.normalized_text
    assert raw[output.offset_map[output.normalized_text.index("5.5")]] == "5"


def test_zero_width_joiners_are_removed_without_destroying_indic_scripts():
    output = ClinicalPreprocessor().run("बुख\u200cार है। ಜ್ವ\u200dರ ಇದೆ.")
    assert "\u200c" not in output.normalized_text and "\u200d" not in output.normalized_text
    assert "बुखार" in output.normalized_text and "ಜ್ವರ" in output.normalized_text


def test_negated_facts_never_appear_in_present_hpi_and_empty_record_is_explicit(pipeline):
    result = pipeline.run("I do not have diabetes, but I have fever for 5 days.")
    assert all("diabetes" not in item.lower() for item in result.patient_record.sections.get("HPI", []))
    assert "PERTINENT NEGATIVES" in result.summary
    empty = pipeline.run("unstructured words only")
    assert "Insufficient data" in empty.summary
    assert "Decision support only" in empty.summary
