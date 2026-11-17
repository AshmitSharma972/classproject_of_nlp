"""Regression tests for single-display output of repeated clinical mentions."""
from src.pipeline import ClinicalPipeline
from web_api import _summary_items


def test_repeated_present_concept_is_shown_once_in_summary_and_summary_items():
    result = ClinicalPipeline().run("I have fever. Fever is still present. I have fever.")

    visible_fever = [
        item for values in result.patient_record.sections.values() for item in values
        if item.casefold() == "fever"
    ]
    assert visible_fever == ["fever"]

    summary_fever = [item for item in _summary_items(result.patient_record) if item["text"].casefold() == "fever"]
    assert len(summary_fever) == 1
    assert len(summary_fever[0]["evidence"]) == 3


def test_same_word_with_different_assertion_is_not_collapsed():
    result = ClinicalPipeline().run("I have fever. I do not have fever.")
    facts = [item for values in result.patient_record.sections.values() for item in values]
    assert any(item.casefold() == "fever" for item in facts)
    assert any(item.casefold() == "no fever (negated)" for item in facts)


def test_summary_template_defensively_deduplicates_visible_bullets():
    from src.summarization.templates import format_bullet_list
    bullets = format_bullet_list(["fever", "Fever", " fever "])
    assert bullets.casefold().count("fever") == 1


def test_same_concept_with_duration_is_displayed_once_in_the_full_summary():
    result = ClinicalPipeline().run("I have fever. Fever for 5 days. I have fever.")
    summary = result.summary.casefold()
    assert summary.count("fever") == 1
    assert "5 days" in summary
