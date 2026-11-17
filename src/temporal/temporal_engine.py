"""
Temporal Engine for the clinical NLP pipeline.

Extracts and normalizes temporal expressions from preprocessing output,
builds a clinical timeline, and resolves simple anaphora.

INVARIANTS:
- All TimeExpr spans reference ORIGINAL raw text offsets.
- Temporal information is never invented; only source-text-supported
  expressions are emitted.
- Provenance is preserved for every interpretation.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from src.common.schemas import (
    Assertion, AssertionStatus, Entity, EntityType, Experiencer, PreprocessingOutput,
    SentenceInfo, Span, TimeExpr, TimeExprType, TimelineEntry,
)
from src.temporal.patterns import (
    ANAPHORA_PATTERNS, DATE_PATTERNS, DURATION_PATTERNS,
    EPISODE_PATTERNS, ONSET_PATTERNS, RELATIVE_PATTERNS,
    SIMPLE_DURATION_PATTERN, normalize_duration,
)
from src.utils import get_logger

logger = get_logger("TemporalEngine")


class TemporalEngine:
    """
    Extracts temporal expressions from preprocessed text, normalizes them,
    resolves anaphoric references, and builds a clinical timeline.
    """

    def __init__(self, encounter_date: Optional[date] = None):
        self.encounter_date = encounter_date

    def run(
        self,
        entities: List[Entity],
        assertions: List[Assertion],
        preprocessing_output: PreprocessingOutput,
    ) -> Tuple[List[TimeExpr], List[TimelineEntry]]:
        """
        Main entry point.

        Args:
            entities: NER output entities.
            assertions: Assertion engine output.
            preprocessing_output: Full preprocessing output including sentences.

        Returns:
            (time_expressions, timeline)
        """
        logger.info("TemporalEngine: starting temporal extraction")

        raw_text = preprocessing_output.raw_text
        sentences = preprocessing_output.sentences

        time_exprs: List[TimeExpr] = []
        norm_text = preprocessing_output.normalized_text
        offset_map = preprocessing_output.offset_map
        cursor = 0

        for sent in sentences:
            sent_text = sent.text
            raw_base = sent.raw_start
            sent_norm_start = norm_text.find(sent.text, cursor) if norm_text else 0
            if sent_norm_start == -1:
                sent_norm_start = cursor
            else:
                cursor = sent_norm_start + len(sent.text)

            # 1. Simple durations (highest priority -- match full phrase first)
            for m in SIMPLE_DURATION_PATTERN.finditer(sent_text):
                qty = m.group(1)
                unit = m.group(2)
                norm = normalize_duration(qty, unit)
                te = self._make_time_expr(
                    m, raw_base, sent.sentence_id,
                    TimeExprType.DURATION, norm,
                    offset_map, sent_norm_start, raw_text,
                )
                if not self._overlaps_existing(te, time_exprs):
                    time_exprs.append(te)

            # 2. Duration patterns (with context words like "for", "since")
            for pat in DURATION_PATTERNS:
                for m in pat.finditer(sent_text):
                    te = self._make_time_expr(
                        m, raw_base, sent.sentence_id,
                        TimeExprType.DURATION,
                        self._normalize_from_groups(m),
                        offset_map, sent_norm_start, raw_text,
                    )
                    if not self._overlaps_existing(te, time_exprs):
                        time_exprs.append(te)

            # 3. Onset patterns
            for pat in ONSET_PATTERNS:
                for m in pat.finditer(sent_text):
                    te = self._make_time_expr(
                        m, raw_base, sent.sentence_id,
                        TimeExprType.ONSET,
                        self._normalize_from_groups(m),
                        offset_map, sent_norm_start, raw_text,
                    )
                    if not self._overlaps_existing(te, time_exprs):
                        time_exprs.append(te)

            # 4. Relative date patterns
            for pat in RELATIVE_PATTERNS:
                for m in pat.finditer(sent_text):
                    te = self._make_time_expr(
                        m, raw_base, sent.sentence_id,
                        TimeExprType.RELATIVE,
                        self._normalize_relative(m.group(0)),
                        offset_map, sent_norm_start, raw_text,
                    )
                    if not self._overlaps_existing(te, time_exprs):
                        time_exprs.append(te)

            # 5. Absolute date patterns
            for pat in DATE_PATTERNS:
                for m in pat.finditer(sent_text):
                    te = self._make_time_expr(
                        m, raw_base, sent.sentence_id,
                        TimeExprType.DATE,
                        m.group(0).lower(),
                        offset_map, sent_norm_start, raw_text,
                    )
                    if not self._overlaps_existing(te, time_exprs):
                        time_exprs.append(te)

            # 6. Previous episode cues
            for pat in EPISODE_PATTERNS:
                for m in pat.finditer(sent_text):
                    te = self._make_time_expr(
                        m, raw_base, sent.sentence_id,
                        TimeExprType.PREVIOUS_EPISODE,
                        m.group(0).lower(),
                        offset_map, sent_norm_start, raw_text,
                    )
                    if not self._overlaps_existing(te, time_exprs):
                        time_exprs.append(te)

        # 7. Link temporal expressions to nearby entities (same sentence)
        self._link_to_entities(time_exprs, entities, assertions, raw_text, sentences)

        # 8. Resolve anaphora
        self._resolve_anaphora(time_exprs, entities, assertions, sentences, preprocessing_output)

        # 9. Build timeline
        timeline = self._build_timeline(time_exprs, entities)

        logger.info(
            "TemporalEngine: extracted %d temporal expressions, %d timeline entries",
            len(time_exprs), len(timeline),
        )
        return time_exprs, timeline

    def _normalize_relative(self, value: str) -> str:
        value = value.lower().strip()
        if self.encounter_date and value.endswith("yesterday"):
            return (self.encounter_date - timedelta(days=1)).isoformat()
        if self.encounter_date and value == "today":
            return self.encounter_date.isoformat()
        if "childhood" in value:
            return "childhood"
        if "birth" in value:
            return "birth"
        return value

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _make_time_expr(
        self,
        match: re.Match,
        raw_base: int,
        sentence_id: int,
        te_type: TimeExprType,
        normalized: Optional[str],
        offset_map: Optional[Dict[int, int]] = None,
        sent_norm_start: int = 0,
        raw_text: str = "",
    ) -> TimeExpr:
        """Build a TimeExpr from a regex match, anchored to original raw text."""
        if offset_map:
            raw_start = offset_map.get(sent_norm_start + match.start(), raw_base + match.start())
            raw_end = offset_map.get(sent_norm_start + match.end() - 1, raw_base + match.end() - 1) + 1
        else:
            raw_start = raw_base + match.start()
            raw_end = raw_base + match.end()

        if raw_text and 0 <= raw_start < raw_end <= len(raw_text):
            raw_text_fragment = raw_text[raw_start:raw_end]
        else:
            raw_text_fragment = match.group(0)

        span = Span(
            text=raw_text_fragment,
            start=raw_start,
            end=raw_end,
            source_sentence_id=sentence_id,
        )
        return TimeExpr(
            span=span,
            type=te_type,
            raw_value=raw_text_fragment,
            normalized_value=normalized,
            linked_entity_ids=[],
        )

    def _normalize_from_groups(self, match: re.Match) -> Optional[str]:
        """Try to normalize using captured groups (quantity + unit)."""
        try:
            groups = [g for g in match.groups() if g is not None]
            if len(groups) >= 2:
                return normalize_duration(groups[-2], groups[-1])
            if groups and groups[0].lower() in ("childhood", "birth"):
                return groups[0].lower()
        except Exception:
            pass
        return match.group(0).lower()

    def _overlaps_existing(self, te: TimeExpr, existing: List[TimeExpr]) -> bool:
        """Return True if te overlaps any span in existing (prevents duplicates)."""
        for ex in existing:
            if not (te.span.end <= ex.span.start or te.span.start >= ex.span.end):
                return True
        return False

    def _link_to_entities(self, time_exprs, entities, assertions, raw_text, sentences) -> None:
        """
        Link only to a preceding patient symptom/disease in the same clause.
        """
        assertion_by_id = {a.entity_id: a for a in assertions}
        sentence_by_id = {s.sentence_id: s for s in sentences}
        terminator = re.compile(r"\b(?:but|however|although)\b|[;.!?]", re.I)
        for te in time_exprs:
            sent = sentence_by_id.get(te.span.source_sentence_id)
            if sent is None:
                continue
            left, right = sent.raw_start, sent.raw_end
            for m in terminator.finditer(raw_text[sent.raw_start:sent.raw_end]):
                start, end = sent.raw_start + m.start(), sent.raw_start + m.end()
                if end <= te.span.start:
                    left = end
                elif start >= te.span.end:
                    right = start
                    break
            eligible = [e for e in entities if e.span.source_sentence_id == te.span.source_sentence_id
                        and e.type in (EntityType.SYMPTOM, EntityType.DISEASE)
                        and left <= e.span.start < te.span.start and e.span.end <= right
                        and (e.id not in assertion_by_id or
                             (assertion_by_id[e.id].status not in (AssertionStatus.NEGATED, AssertionStatus.FAMILY)
                              and assertion_by_id[e.id].experiencer != Experiencer.OTHER))]
            if not eligible and te.type in (TimeExprType.RELATIVE, TimeExprType.DATE):
                eligible = [e for e in entities if e.span.source_sentence_id == te.span.source_sentence_id
                            and e.type in (EntityType.SYMPTOM, EntityType.DISEASE)
                            and left <= e.span.start and e.span.end <= right
                            and (e.id not in assertion_by_id or
                                 (assertion_by_id[e.id].status not in (AssertionStatus.NEGATED, AssertionStatus.FAMILY)
                                  and assertion_by_id[e.id].experiencer != Experiencer.OTHER))]
                eligible.sort(key=lambda e: abs(e.span.start - te.span.end))
                eligible = eligible[:1]
            if not eligible:
                continue
            eligible.sort(key=lambda e: e.span.end)
            chosen = [eligible[-1]]
            if len(eligible) > 1:
                between = raw_text[eligible[-2].span.end:eligible[-1].span.start]
                if re.fullmatch(r"\s*(?:,\s*)?(?:and|&)\s*", between, re.I):
                    chosen.insert(0, eligible[-2])
            te.linked_entity_ids = [e.id for e in chosen]

    def _resolve_anaphora(
        self,
        time_exprs: List[TimeExpr],
        entities: List[Entity],
        assertions: List[Assertion],
        sentences: List[SentenceInfo],
        preprocessing_output: PreprocessingOutput,
    ) -> None:
        """
        Simple anaphora resolution:
        - "similar problem" / "last time" / "same symptoms" → links to
          the most recently seen SYMPTOM/DISEASE entity in earlier sentences.
        - "the fever" / "this cough" → links to the most recent matching entity.
        """
        raw_text = preprocessing_output.raw_text
        assertion_by_id = {a.entity_id: a for a in assertions}

        # Build a dict: sentence_id → list of entities (sorted by appearance)
        entities_by_sent: Dict[int, List[Entity]] = {}
        for ent in entities:
            sid = ent.span.source_sentence_id
            entities_by_sent.setdefault(sid, []).append(ent)

        for sent in sentences:
            sent_text = sent.text
            raw_base = sent.raw_start

            # Check anaphora patterns
            for pat in ANAPHORA_PATTERNS:
                for m in pat.finditer(sent_text):
                    # Get the noun being referred to (last group)
                    noun = m.group(1).lower() if m.lastindex else m.group(0).lower()
                    # Find most recent matching entity in earlier sentences
                    target_entity = self._find_recent_entity_by_noun(
                        noun, sent.sentence_id, entities_by_sent, assertion_by_id
                    )
                    if target_entity is None:
                        continue

                    # Check if there's already a time_expr for this anaphora span
                    raw_start = raw_base + m.start()
                    raw_end = raw_base + m.end()
                    existing_te = next(
                        (t for t in time_exprs
                         if t.span.start == raw_start and t.span.end == raw_end),
                        None
                    )
                    if existing_te:
                        if target_entity.id not in existing_te.linked_entity_ids:
                            existing_te.linked_entity_ids.append(target_entity.id)
                    # (No new TimeExpr for anaphora -- just linkage)

    def _find_recent_entity_by_noun(
        self,
        noun: str,
        current_sentence_id: int,
        entities_by_sent: Dict[int, List[Entity]],
        assertions_by_id: Dict[str, Assertion],
    ) -> Optional[Entity]:
        """Return the most recently seen entity whose text matches noun."""
        for sid in range(current_sentence_id - 1, -1, -1):
            for ent in reversed(entities_by_sent.get(sid, [])):
                assertion = assertions_by_id.get(ent.id)
                if (noun.lower() in ent.span.text.lower()
                    and ent.type in (EntityType.SYMPTOM, EntityType.DISEASE)
                    and (assertion is None or assertion.status not in (AssertionStatus.NEGATED, AssertionStatus.FAMILY))
                    and (assertion is None or assertion.experiencer != Experiencer.OTHER)):
                    return ent
        return None

    def _build_timeline(
        self,
        time_exprs: List[TimeExpr],
        entities: List[Entity],
    ) -> List[TimelineEntry]:
        """
        Construct a simple ordered timeline from temporal expressions.

        Ordering heuristic:
        1. Absolute dates (DATE) come first, sorted lexicographically.
        2. HISTORICAL before ONSET before DURATION.
        3. PREVIOUS_EPISODE before current events.
        4. RELATIVE expressions at the end.

        Only time expressions with linked entities are included in the timeline.
        """
        ORDER_MAP = {
            TimeExprType.DATE: 0,
            TimeExprType.PREVIOUS_EPISODE: 1,
            TimeExprType.ONSET: 2,
            TimeExprType.DURATION: 3,
            TimeExprType.TREATMENT_DATE: 4,
            TimeExprType.PROCEDURE_DATE: 5,
            TimeExprType.RELATIVE: 6,
        }

        # Build entity id → text lookup
        entity_text: Dict[str, str] = {e.id: e.span.text for e in entities}

        timeline_entries: List[TimelineEntry] = []
        for te in time_exprs:
            if not te.linked_entity_ids:
                continue
            linked_texts = [
                entity_text.get(eid, "?") for eid in te.linked_entity_ids
            ]
            if linked_texts:
                label = f"{', '.join(linked_texts)} -- {te.normalized_value or te.raw_value}"
            else:
                label = te.normalized_value or te.raw_value

            order = ORDER_MAP.get(te.type, 99)
            timeline_entries.append(
                TimelineEntry(
                    order=order,
                    label=label,
                    time_expr_id=te.id,
                    linked_entity_ids=list(te.linked_entity_ids),
                    raw_text=te.raw_value,
                )
            )

        timeline_entries.sort(key=lambda t: (t.order, t.label))
        # Re-number orders 0, 1, 2, ... for clean output
        for idx, entry in enumerate(timeline_entries):
            entry.order = idx

        return timeline_entries
