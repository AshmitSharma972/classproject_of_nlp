"""
Patient Record Builder for the clinical NLP pipeline.

Constructs a single unified PatientRecord from all pipeline stage outputs.
Every clinically meaningful item retains source provenance.

INVARIANTS:
- No data is lost between stages.
- DB-ready conceptually -- PatientRecord serializes cleanly to JSON.
- No FHIR or database integration in v1.
- No orchestration logic here -- only record assembly.
"""

from __future__ import annotations

import uuid
import re
from datetime import datetime
from typing import Any, Dict, List

from src.common.schemas import (
    Assertion, AssertionStatus, Entity, EntityType,
    PatientRecord, PreprocessingOutput, Relation,
    SectionItem, SectionType, TimeExpr, TimelineEntry,
)
from src.utils import get_logger

logger = get_logger("PatientRecordBuilder")


class PatientRecordBuilder:
    """Assembles a PatientRecord from all pipeline stage outputs."""

    def build(
        self,
        raw_text: str,
        preprocessing_output: PreprocessingOutput,
        entities: List[Entity],
        assertions: List[Assertion],
        time_expressions: List[TimeExpr],
        relations: List[Relation],
        section_items: List[SectionItem],
        timeline: List[TimelineEntry],
    ) -> PatientRecord:
        """
        Build and return a complete PatientRecord.

        Args:
            raw_text: Original unmodified patient narration.
            preprocessing_output: Output of the preprocessing stage.
            entities: NER-extracted entities.
            assertions: Assertion/negation tags.
            time_expressions: Temporal expressions.
            relations: Extracted entity relations.
            section_items: Section-classified items.
            timeline: Ordered clinical timeline.

        Returns:
            A fully populated, JSON-serializable PatientRecord.
        """
        # Build grouped sections dict for easy rendering
        sections_grouped = self._group_sections(section_items, entities, assertions)

        # Build preprocessing metadata
        prep_meta: Dict[str, Any] = {
            "sentence_count": len(preprocessing_output.sentences),
            "token_count": preprocessing_output.metadata.get("token_count", 0),
            "spell_flag_count": len(preprocessing_output.spell_flags),
            "spell_flags": preprocessing_output.spell_flags,
        }

        record = PatientRecord(
            id=str(uuid.uuid4()),
            raw_text=raw_text,
            created_at=datetime.utcnow(),
            entities=entities,
            assertions=assertions,
            time_expressions=time_expressions,
            relations=relations,
            section_items=section_items,
            timeline=timeline,
            sections=sections_grouped,
            preprocessing_metadata=prep_meta,
        )

        logger.info(
            "PatientRecordBuilder: record %s built -- %d entities, %d assertions, "
            "%d temporal, %d relations, %d section items, %d timeline entries",
            record.id,
            len(entities), len(assertions), len(time_expressions),
            len(relations), len(section_items), len(timeline),
        )
        return record

    def _group_sections(
        self, section_items: List[SectionItem], entities: List[Entity], assertions: List[Assertion]
    ) -> Dict[str, List[str]]:
        """
        Build a dict mapping section names to their display text list.
        Preserves insertion order (entities are already sorted by appearance).
        """
        grouped: Dict[str, List[str]] = {s.value: [] for s in SectionType}
        entity_by_id = {entity.id: entity for entity in entities}
        assertion_by_entity_id = {assertion.entity_id: assertion for assertion in assertions}
        # Raw entities keep every source mention. For output, a concept key
        # acts like a set: fever, Fever, and fever with the same assertion but
        # different duration wording become one visible clinical fact.
        # Assertion status stays in the key so "fever" and "No fever" are
        # never incorrectly merged.
        output_location_by_key: Dict[tuple[str, str, str], tuple[str, int]] = {}

        for item in section_items:
            key = item.section.value
            display = item.display_text or item.provenance_span.text
            entity = entity_by_id.get(item.entity_id)
            assertion = assertion_by_entity_id.get(item.entity_id)
            concept = (entity.normalized_text if entity and entity.normalized_text else item.provenance_span.text)
            concept_key = re.sub(r"\s+", " ", concept).strip().casefold()
            status_key = assertion.status.value if assertion else "PRESENT"
            experiencer_key = assertion.experiencer.value if assertion else "PATIENT"
            output_key = (concept_key, status_key, experiencer_key)
            if output_key not in output_location_by_key:
                grouped[key].append(display)
                output_location_by_key[output_key] = (key, len(grouped[key]) - 1)
            else:
                # Prefer the display that retains useful linked information,
                # for example "fever (5 days)" over simply "fever".
                existing_section, existing_index = output_location_by_key[output_key]
                if len(display) > len(grouped[existing_section][existing_index]):
                    grouped[existing_section][existing_index] = display

        # Remove empty sections
        return {k: v for k, v in grouped.items() if v}
