"""
Section Classifier for the clinical NLP pipeline.

Assigns each entity (with its assertion and temporal context) to one of the
8 standard clinical history sections, using deterministic rule-based logic.

INVARIANTS:
- Family-history facts never become patient-history facts.
- Negated facts retain their NEGATED status within their section.
- All section items retain full provenance.
- No orchestration logic here -- only classification.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from src.common.schemas import (
    Assertion, AssertionStatus, Entity, EntityType, Relation, SectionItem,
    SectionType, Span, TimeExpr,
)
from src.section_classification.rules import classify_entity
from src.utils import get_logger

logger = get_logger("SectionClassifier")


class SectionClassifier:
    """
    Classifies entities into clinical note sections.

    Uses deterministic rules first.
    An ML classifier hook is included but not wired (see _ml_classify).
    """

    def run(
        self,
        entities: List[Entity],
        assertions: List[Assertion],
        time_expressions: List[TimeExpr],
        relations: List[Relation],
    ) -> List[SectionItem]:
        """
        Classify all entities into sections.

        Args:
            entities: All named entities (from NER stage).
            assertions: One assertion per entity (from Assertion stage).
            time_expressions: All temporal expressions (from Temporal stage).
            relations: All extracted relations (from Relation stage).

        Returns:
            List of SectionItem objects with full provenance.
        """
        # Build quick lookups
        assertion_map: Dict[str, Assertion] = {
            a.entity_id: a for a in assertions
        }
        # Map entity_id → list of TimeExprs linked to it
        te_map: Dict[str, List[TimeExpr]] = {}
        for te in time_expressions:
            for eid in te.linked_entity_ids:
                te_map.setdefault(eid, []).append(te)

        # Map entity_id → list of relation ids that involve it
        rel_map: Dict[str, List[str]] = {}
        for rel in relations:
            rel_map.setdefault(rel.head_entity_id, []).append(rel.id)
            rel_map.setdefault(rel.tail_entity_id, []).append(rel.id)

        section_items: List[SectionItem] = []
        chief_complaint_filled = False

        # Sort entities by their appearance in the text
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.span.source_sentence_id, e.span.start),
        )

        for entity in sorted_entities:
            assertion = assertion_map.get(entity.id)
            linked_tes = te_map.get(entity.id, [])
            linked_rel_ids = rel_map.get(entity.id, [])

            if entity.type in (EntityType.DURATION, EntityType.DATE):
                has_temporal_link = any(
                    t.linked_entity_ids and not (t.span.end <= entity.span.start or entity.span.end <= t.span.start)
                    for t in time_expressions
                )
                if not has_temporal_link:
                    continue
            if entity.type == EntityType.DOSAGE and not linked_rel_ids:
                continue

            # Determine section
            section = classify_entity(
                entity=entity,
                assertion=assertion,
                time_expressions=linked_tes,
                chief_complaint_filled=chief_complaint_filled,
            )

            if section == SectionType.CHIEF_COMPLAINT:
                chief_complaint_filled = True

            # Build display text
            display = self._build_display_text(entity, assertion, linked_tes)

            item = SectionItem(
                id=str(uuid.uuid4()),
                entity_id=entity.id,
                section=section,
                assertion_id=assertion.id if assertion else None,
                temporal_ids=[te.id for te in linked_tes],
                relation_ids=linked_rel_ids,
                provenance_span=entity.span,
                display_text=display,
            )
            section_items.append(item)

        logger.info(
            "SectionClassifier: %d entities -> %d section items across %d sections",
            len(entities), len(section_items),
            len({si.section for si in section_items}),
        )
        return section_items

    def _build_display_text(
        self,
        entity: Entity,
        assertion: Optional[Assertion],
        time_expressions: List[TimeExpr],
    ) -> str:
        """Construct a human-readable label for the section item."""
        text = entity.span.text

        if assertion:
            status = assertion.status
            if status == AssertionStatus.NEGATED:
                text = f"No {text} (negated)"
            elif status == AssertionStatus.POSSIBLE:
                text = f"Possible {text}"
            elif status == AssertionStatus.HISTORICAL:
                text = f"{text} (historical)"
            elif status == AssertionStatus.FAMILY:
                experiencer_note = (
                    f" [{assertion.trigger_text}]"
                    if assertion.trigger_text
                    else " [family]"
                )
                text = f"{text}{experiencer_note}"

        # Append temporal qualifiers
        if time_expressions:
            temporal_strs = [
                te.normalized_value or te.raw_value for te in time_expressions
            ]
            text = f"{text} ({'; '.join(temporal_strs)})"

        return text

    def _ml_classify(self, entity: Entity, context: str) -> Optional[SectionType]:
        """
        Hook for an optional ML/model-based section classifier.

        To implement:
        1. Encode (entity type, assertion status, entity text, sentence context)
           into a feature vector or prompt.
        2. Call a fine-tuned text classifier (e.g. scikit-learn, HuggingFace).
        3. Return the predicted SectionType.

        This method is intentionally not called in production flow;
        wire it in by modifying run() after the rule-based call.
        """
        raise NotImplementedError(
            "ML section classifier not implemented. "
            "Implement this method and call it as a fallback in run()."
        )
