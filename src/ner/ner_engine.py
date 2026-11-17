"""
ner_engine.py -- Rule-Based Clinical Named Entity Recognition Engine
====================================================================

Implements:
  - ``BaseNER``        : Abstract base class all NER back-ends must satisfy.
  - ``ClinicalNER``    : Production rule-based engine using lexicons + regex.
  - ``HFNERAdapter``   : Stub adapter showing how to wire a HuggingFace NER
                         pipeline as a drop-in replacement.

Algorithm (ClinicalNER.run):
  1. For every sentence in ``PreprocessingOutput.sentences``:
       a. Lexicon scan (longest-match wins for overlapping candidates).
       b. Regex scan for DOSAGE, FREQUENCY, DURATION, DATE, TEST_RESULT,
          SEVERITY.
       c. Each match is anchored to the ORIGINAL raw text via
          ``sentence.raw_start + match.start()``.
  2. Global deduplication: when two Entity spans overlap keep the longer one.
  3. Return entities sorted by (source_sentence_id, span.start).

All spans point into the ORIGINAL raw input text -- never into any normalised
or lowercased derivative.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.common.schemas import (
    Entity,
    EntityType,
    PreprocessingOutput,
    SentenceInfo,
    Span,
)
from src.utils import get_logger
from src.ner.lexicons import (
    ALLERGEN_LEXICON,
    BODY_PART_LEXICON,
    COMPILED_DATE,
    COMPILED_DOSAGE,
    COMPILED_DURATION,
    COMPILED_FREQUENCY,
    COMPILED_SEVERITY,
    COMPILED_TEST_RESULT,
    DISEASE_LEXICON,
    FAMILY_MEMBER_LEXICON,
    MEDICATION_LEXICON,
    PROCEDURE_LEXICON,
    SEVERITY_LEXICON,
    SYMPTOM_LEXICON,
    TEST_LEXICON,
    TEST_RESULT_LEXICON,
)

logger = get_logger("ClinicalNER")

# ---------------------------------------------------------------------------
# Confidence constants
# ---------------------------------------------------------------------------
_CONF_LEXICON: float = 1.0
_CONF_REGEX: float = 0.85

# ---------------------------------------------------------------------------
# Lexicon registry: entity-type → (frozenset of lower-case terms)
# ---------------------------------------------------------------------------
_LEXICON_REGISTRY: List[Tuple[EntityType, frozenset]] = [
    # Longer (multi-word) lexicons must be searched first so that the
    # longest-match suppression works correctly within each lexicon.
    (EntityType.SYMPTOM,       SYMPTOM_LEXICON),
    (EntityType.DISEASE,       DISEASE_LEXICON),
    (EntityType.MEDICATION,    MEDICATION_LEXICON),
    (EntityType.BODY_PART,     BODY_PART_LEXICON),
    (EntityType.TEST,          TEST_LEXICON),
    (EntityType.PROCEDURE,     PROCEDURE_LEXICON),
    (EntityType.ALLERGEN,      ALLERGEN_LEXICON),
    (EntityType.FAMILY_MEMBER, FAMILY_MEMBER_LEXICON),
    (EntityType.TEST_RESULT,   TEST_RESULT_LEXICON),
    (EntityType.SEVERITY,      SEVERITY_LEXICON),
]

# ---------------------------------------------------------------------------
# Internal data class for raw span candidates before Entity creation
# ---------------------------------------------------------------------------

class _Candidate:
    """Lightweight span + type holder before Pydantic Entity construction."""

    __slots__ = ("raw_start", "raw_end", "text", "entity_type", "confidence")

    def __init__(
        self,
        raw_start: int,
        raw_end: int,
        text: str,
        entity_type: EntityType,
        confidence: float,
    ) -> None:
        self.raw_start = raw_start
        self.raw_end = raw_end
        self.text = text
        self.entity_type = entity_type
        self.confidence = confidence

    def length(self) -> int:
        return self.raw_end - self.raw_start

    def overlaps(self, other: "_Candidate") -> bool:
        return self.raw_start < other.raw_end and other.raw_start < self.raw_end


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseNER(ABC):
    """Abstract interface for all NER back-ends used in the pipeline."""

    @abstractmethod
    def run(self, preprocessing_output: PreprocessingOutput) -> List[Entity]:
        """
        Run named entity recognition over ``preprocessing_output``.

        Parameters
        ----------
        preprocessing_output:
            The structured output from the preprocessing stage, including
            tokenised sentences with raw-text character offsets.

        Returns
        -------
        List[Entity]
            Entities sorted by (source_sentence_id, span.start).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HuggingFace adapter stub
# ---------------------------------------------------------------------------

class HFNERAdapter(BaseNER):
    """
    Stub adapter for integrating a HuggingFace token-classification pipeline.

    Usage
    -----
    To activate, install ``transformers`` and ``torch`` (or ``tensorflow``),
    then replace this stub with a concrete implementation::

        from transformers import pipeline as hf_pipeline

        class HFNERAdapter(BaseNER):
            def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
                self._pipe = hf_pipeline(
                    "ner",
                    model=model_name,
                    aggregation_strategy="simple",
                )

            def run(self, preprocessing_output: PreprocessingOutput) -> List[Entity]:
                entities: List[Entity] = []
                for sent in preprocessing_output.sentences:
                    results = self._pipe(sent.text)
                    for r in results:
                        raw_s = sent.raw_start + r["start"]
                        raw_e = sent.raw_start + r["end"]
                        span = Span(
                            text=r["word"],
                            start=raw_s,
                            end=raw_e,
                            source_sentence_id=sent.sentence_id,
                        )
                        entities.append(
                            Entity(
                                span=span,
                                type=_map_hf_label(r["entity_group"]),
                                confidence=float(r["score"]),
                            )
                        )
                return entities

    The ``_map_hf_label`` helper would translate HF label strings
    (e.g. ``"B-DISEASE"``) to the project's ``EntityType`` enum values.
    """

    def run(self, preprocessing_output: PreprocessingOutput) -> List[Entity]:
        raise NotImplementedError(
            "HFNERAdapter is a stub. "
            "See class docstring for integration instructions."
        )


# ---------------------------------------------------------------------------
# Main rule-based engine
# ---------------------------------------------------------------------------

class ClinicalNER(BaseNER):
    """
    Lexicon + regex rule-based clinical NER engine.

    Produces ``Entity`` objects with character offsets anchored to the
    ORIGINAL raw input text (``preprocessing_output.raw_text``).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, preprocessing_output: PreprocessingOutput) -> List[Entity]:
        """
        Extract all clinical entities from ``preprocessing_output``.

        Parameters
        ----------
        preprocessing_output:
            Output of the preprocessing stage.

        Returns
        -------
        List[Entity]
            Deduplicated entities sorted by (source_sentence_id, span.start).
        """
        logger.info(
            "ClinicalNER.run -- processing %d sentence(s)",
            len(preprocessing_output.sentences),
        )

        all_candidates: List[_Candidate] = []
        cursor = 0
        norm_text = preprocessing_output.normalized_text
        offset_map = preprocessing_output.offset_map
        raw_text = preprocessing_output.raw_text

        for sent in preprocessing_output.sentences:
            sent_norm_start = norm_text.find(sent.text, cursor) if norm_text else 0
            if sent_norm_start == -1:
                sent_norm_start = cursor
            else:
                cursor = sent_norm_start + len(sent.text)

            sentence_candidates = self._process_sentence(
                sent, offset_map, sent_norm_start, raw_text
            )
            all_candidates.extend(sentence_candidates)

        # Global deduplication (longest-match across the whole document)
        deduplicated = _deduplicate(all_candidates)

        # Build Pydantic Entity objects
        entities: List[Entity] = []
        for cand in deduplicated:
            # Determine which sentence this candidate belongs to.
            # (candidates already carry per-sentence sentence_id via the
            # raw_start, but we need to find it properly)
            sent_id = _find_sentence_id(
                cand.raw_start, preprocessing_output.sentences
            )
            span = Span(
                text=cand.text,
                start=cand.raw_start,
                end=cand.raw_end,
                source_sentence_id=sent_id,
            )
            entity = Entity(
                span=span,
                type=cand.entity_type,
                confidence=cand.confidence,
                normalized_text={"htn": "hypertension", "dm": "diabetes mellitus",
                                 "t2dm": "type 2 diabetes mellitus", "sob": "shortness of breath"}.get(
                                     cand.text.lower(), None),
            )
            entities.append(entity)

        # Sort by (sentence_id, span.start)
        entities.sort(key=lambda e: (e.span.source_sentence_id, e.span.start))

        logger.info("ClinicalNER.run -- extracted %d entities", len(entities))
        return entities

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_sentence(
        self,
        sent: SentenceInfo,
        offset_map: Dict[int, int],
        sent_norm_start: int,
        raw_text: str,
    ) -> List[_Candidate]:
        """Return all candidates extracted from a single sentence."""
        candidates: List[_Candidate] = []

        # ---- 1. Lexicon scan (multi-word longest-match) ----------------
        candidates.extend(self._lexicon_scan(sent, offset_map, sent_norm_start, raw_text))

        # ---- 2. Regex scan for pattern-based entity types -------------
        candidates.extend(self._regex_scan(sent, offset_map, sent_norm_start, raw_text))

        return candidates

    # ------------------------------------------------------------------
    # Lexicon scanning
    # ------------------------------------------------------------------

    def _lexicon_scan(
        self,
        sent: SentenceInfo,
        offset_map: Dict[int, int],
        sent_norm_start: int,
        raw_text: str,
    ) -> List[_Candidate]:
        """
        Scan ``sent.text`` against all lexicons using a sliding-window
        approach that prefers the *longest* matching term when multiple
        candidate terms share the same start offset.

        Offsets in candidates are absolute positions in the original raw text.
        """
        text_lower = sent.text.lower()
        text_len = len(text_lower)
        candidates: List[_Candidate] = []

        for entity_type, lexicon in _LEXICON_REGISTRY:
            # Sort terms longest-first so that longer matches shadow shorter
            # ones at the same position during the add phase.
            sorted_terms = sorted(lexicon, key=len, reverse=True)

            for term in sorted_terms:
                term_len = len(term)
                start = 0
                while start <= text_len - term_len:
                    idx = text_lower.find(term, start)
                    if idx == -1:
                        break

                    end = idx + term_len

                    # Word-boundary guard: ensure we don't match partial words.
                    # Check that the character immediately before/after is not
                    # alphanumeric.
                    if not _is_word_boundary(text_lower, idx, end):
                        start = idx + 1
                        continue

                    norm_s = sent_norm_start + idx
                    norm_e = sent_norm_start + end
                    raw_s = offset_map.get(norm_s, sent.raw_start + idx) if offset_map else (sent.raw_start + idx)
                    raw_e = (offset_map.get(norm_e - 1, sent.raw_start + end - 1) + 1) if offset_map else (sent.raw_start + end)

                    if raw_text and 0 <= raw_s < raw_e <= len(raw_text):
                        original_match = raw_text[raw_s:raw_e]
                    else:
                        original_match = sent.text[idx:end]

                    candidates.append(
                        _Candidate(
                            raw_start=raw_s,
                            raw_end=raw_e,
                            text=original_match,
                            entity_type=entity_type,
                            confidence=_CONF_LEXICON,
                        )
                    )
                    start = end  # skip past current match to avoid re-matching

        return candidates

    # ------------------------------------------------------------------
    # Regex scanning
    # ------------------------------------------------------------------

    def _regex_scan(
        self,
        sent: SentenceInfo,
        offset_map: Dict[int, int],
        sent_norm_start: int,
        raw_text: str,
    ) -> List[_Candidate]:
        """Match DOSAGE, FREQUENCY, DURATION, DATE patterns against sentence."""
        candidates: List[_Candidate] = []
        text = sent.text  # keep original casing for verbatim capture

        # Vital-sign and laboratory values are results, not drug dosages.
        measurement_patterns = (
            r"\b\d{2,3}/\d{2,3}(?:\s*mmhg)?\b",
            r"\b\d+(?:\.\d+)?\s*g/dl\b",
            r"\b\d{2,3}(?:\.\d+)?\s*(?:(?:°\s*)?[cf]|degrees?\s*[cf]|bpm|beats/min|%|mg/dl|mmol/l)(?!\w)",
        )
        for pattern in measurement_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                candidates.append(_make_regex_candidate(
                    m, sent, offset_map, sent_norm_start, raw_text, EntityType.TEST_RESULT
                ))

        # --- DOSAGE ---
        for m in COMPILED_DOSAGE.finditer(text):
            candidates.append(_make_regex_candidate(
                m, sent, offset_map, sent_norm_start, raw_text, EntityType.DOSAGE
            ))

        # --- FREQUENCY ---
        for pattern in COMPILED_FREQUENCY:
            for m in pattern.finditer(text):
                candidates.append(_make_regex_candidate(
                    m, sent, offset_map, sent_norm_start, raw_text, EntityType.FREQUENCY
                ))

        # --- DURATION ---
        for pattern in COMPILED_DURATION:
            for m in pattern.finditer(text):
                candidates.append(_make_regex_candidate(
                    m, sent, offset_map, sent_norm_start, raw_text, EntityType.DURATION
                ))

        # --- DATE ---
        for pattern in COMPILED_DATE:
            for m in pattern.finditer(text):
                candidates.append(_make_regex_candidate(
                    m, sent, offset_map, sent_norm_start, raw_text, EntityType.DATE
                ))

        # --- TEST_RESULT (regex -- catches multi-word variants) ---
        for m in COMPILED_TEST_RESULT.finditer(text):
            candidates.append(_make_regex_candidate(
                m, sent, offset_map, sent_norm_start, raw_text, EntityType.TEST_RESULT
            ))

        # --- SEVERITY (regex -- catches pain-scale "X/10" + descriptors) ---
        for m in COMPILED_SEVERITY.finditer(text):
            candidates.append(_make_regex_candidate(
                m, sent, offset_map, sent_norm_start, raw_text, EntityType.SEVERITY
            ))

        return candidates


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_regex_candidate(
    match: re.Match,  # type: ignore[type-arg]
    sent: SentenceInfo,
    offset_map: Dict[int, int],
    sent_norm_start: int,
    raw_text: str,
    entity_type: EntityType,
) -> _Candidate:
    """Build a ``_Candidate`` from a regex match object."""
    norm_s = sent_norm_start + match.start()
    norm_e = sent_norm_start + match.end()
    raw_s = offset_map.get(norm_s, sent.raw_start + match.start()) if offset_map else (sent.raw_start + match.start())
    raw_e = (offset_map.get(norm_e - 1, sent.raw_start + match.end() - 1) + 1) if offset_map else (sent.raw_start + match.end())

    if raw_text and 0 <= raw_s < raw_e <= len(raw_text):
        match_text = raw_text[raw_s:raw_e]
    else:
        match_text = sent.text[match.start():match.end()]

    return _Candidate(
        raw_start=raw_s,
        raw_end=raw_e,
        text=match_text,
        entity_type=entity_type,
        confidence=_CONF_REGEX,
    )


def _is_word_boundary(text: str, start: int, end: int) -> bool:
    """
    Return True if the substring [start, end) is delimited by
    non-alphanumeric / non-hyphen characters (or string boundaries).
    """
    before_ok = (start == 0) or (not text[start - 1].isalnum() and text[start - 1] not in ("-", "_"))
    after_ok  = (end == len(text)) or (not text[end].isalnum() and text[end] not in ("-", "_"))
    return before_ok and after_ok


def _deduplicate(candidates: List[_Candidate]) -> List[_Candidate]:
    """
    Remove overlapping candidates using a greedy longest-match strategy.

    Procedure:
      1. Sort by (raw_start ascending, length descending).
      2. Iterate; accept a candidate only if it does not overlap any
         already-accepted candidate.
    """
    # Sort: earlier start first; for same start, longer span first
    sorted_cands = sorted(
        candidates,
        key=lambda c: (c.raw_start, -c.length()),
    )

    accepted: List[_Candidate] = []
    for cand in sorted_cands:
        overlap = any(cand.overlaps(a) for a in accepted)
        if not overlap:
            accepted.append(cand)

    return accepted


def _find_sentence_id(raw_start: int, sentences: List[SentenceInfo]) -> int:
    """
    Return the ``sentence_id`` of the sentence whose character range
    contains ``raw_start``.  Falls back to 0 if no match is found.
    """
    for sent in sentences:
        if sent.raw_start <= raw_start < sent.raw_end:
            return sent.sentence_id
    # Edge-case: raw_start exactly at last sentence end
    if sentences:
        return sentences[-1].sentence_id
    return 0
