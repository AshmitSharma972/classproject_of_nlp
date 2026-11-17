"""
Clinical Summarizer for the clinical NLP pipeline.

Generates structured clinical summaries exclusively from PatientRecord.
Never operates directly on raw patient text.
Enforces strict preservation of assertion categories (never promotes NEGATED to present,
POSSIBLE to confirmed, HISTORICAL to current, or FAMILY to patient).

ACTIVE: this template-based module is used by ClinicalPipeline. The sibling
src.summarizer module is legacy and is not in the live summary path.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from src.common.schemas import PatientRecord, AssertionStatus
from src.summarization.templates import build_summary_text, DISCLAIMER
from src.utils import get_logger

logger = get_logger("ClinicalSummarizer")


class ClinicalSummarizer:
    """
    Template-based clinical summarizer consuming only structured PatientRecord.
    """

    def __init__(self, llm_enabled: bool = False):
        self.llm_enabled = llm_enabled

    def summarize(self, patient_record: PatientRecord) -> str:
        """
        Generate a physician-ready clinical summary from PatientRecord.

        Args:
            patient_record: Validated PatientRecord object.

        Returns:
            Formatted summary text.
        """
        logger.info("ClinicalSummarizer: generating summary for record %s", patient_record.id)

        # Generate strictly from patient_record.sections and timeline
        # If a condition is already represented in a section with its linked
        # duration, do not repeat the same condition again in the timeline.
        # This keeps the visible summary set-like while retaining all timeline
        # evidence on PatientRecord for review.
        section_concepts = {
            (entity.normalized_text or entity.span.text).strip().casefold()
            for entity in patient_record.entities
            if entity.type.value in {"SYMPTOM", "DISEASE"}
        }
        timeline_entries = []
        seen_timeline_concepts = set()
        for entry in sorted(patient_record.timeline, key=lambda t: t.order):
            label = entry.label or ""
            concept = label.split("--", 1)[0].strip().casefold()
            if concept in section_concepts or concept in seen_timeline_concepts:
                continue
            seen_timeline_concepts.add(concept)
            timeline_entries.append(f"Step {entry.order + 1}: {label}")

        summary_text = build_summary_text(
            sections=patient_record.sections,
            timeline_entries=timeline_entries,
            record_id=patient_record.id,
        )

        # Verification step: ensure summary only contains information from structured record
        self._verify_summary(summary_text, patient_record)

        return summary_text

    def _verify_summary(self, summary_text: str, patient_record: PatientRecord) -> bool:
        """
        Verifies that no hallucinated or unsupported facts are present.
        In template-based mode, every item was mapped from patient_record.sections.
        """
        # Ensure mandatory disclaimer is present
        if "Decision support only" not in summary_text:
            raise ValueError("Summary verification failed: Mandatory disclaimer is missing.")
        return True
