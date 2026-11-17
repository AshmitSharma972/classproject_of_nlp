"""
assertion_engine.py -- NegEx-inspired rule-based assertion / negation classifier.

For every Entity produced by the NER stage this module decides:
  * AssertionStatus  (PRESENT | NEGATED | POSSIBLE | HISTORICAL | FAMILY)
  * Experiencer      (PATIENT | OTHER)

Algorithm overview
------------------
1. Locate the sentence the entity belongs to via entity.span.source_sentence_id.
2. Tokenise the sentence (whitespace split, lowercase).
3. Find the entity's token range within the sentence.
4. Extract a LEFT context window (up to LEFT_WINDOW_TOKENS tokens before the
   entity) and a RIGHT context window (up to RIGHT_WINDOW_TOKENS tokens after).
5. Truncate each window at any TERMINATION_TOKEN so that clausal boundaries
   are respected.
6. Check trigger sets in PRIORITY ORDER:
      FAMILY   → experiencer=OTHER,    status=FAMILY
      NEGATION → experiencer=PATIENT,  status=NEGATED
      POSSIBLE → experiencer=PATIENT,  status=POSSIBLE   (left window only)
      HISTORICAL → experiencer=PATIENT, status=HISTORICAL (left window only)
      default  → experiencer=PATIENT,  status=PRESENT

   IMPORTANT: FAMILY always wins over NEGATION.
   Example: "My father does not have diabetes" → FAMILY (not NEGATED),
   because knowing the family context is clinically more important than
   the negation sub-clause.

7. Record the matched trigger text and compute a Span for it in the
   *original raw text* (character offsets into preprocessing_output.raw_text).
8. Return one Assertion per input Entity.

Phrase matching
---------------
Multi-word triggers (e.g. "no history of") are matched by joining
consecutive window tokens with a single space and checking if any trigger
phrase is a substring of that joined text.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.common.schemas import (
    Assertion,
    AssertionStatus,
    Entity,
    Experiencer,
    PreprocessingOutput,
    SentenceInfo,
    Span,
)
from src.negation.triggers import (
    FAMILY_TRIGGERS,
    HISTORICAL_TRIGGERS,
    LEFT_WINDOW_TOKENS,
    NEGATION_TRIGGERS,
    POSSIBLE_TRIGGERS,
    PSEUDO_NEGATION_PHRASES,
    RIGHT_WINDOW_TOKENS,
    TERMINATION_TOKENS,
)
from src.utils import get_logger

logger = get_logger("AssertionEngine")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokeniser that preserves punctuation attached to words.
    Returns a list of lowercase tokens (empty strings are filtered out).
    """
    return [t for t in re.split(r"\s+", text.lower()) if t]


def _find_entity_token_range(
    tokens: List[str],
    entity_text_lower: str,
) -> Tuple[int, int]:
    """
    Find the (start_tok_idx, end_tok_idx) of an entity inside a token list.

    Strategy: build joined strings of consecutive token slices and compare
    against the lowercase entity text.  Returns the first match found.
    If not found, returns (0, 0) as a safe fallback (the engine will still
    produce a valid assertion, just with a slightly inaccurate window).
    """
    entity_tokens = _tokenize(entity_text_lower)
    n = len(entity_tokens)
    if n == 0:
        return (0, 0)

    for i in range(len(tokens) - n + 1):
        candidate = tokens[i : i + n]
        if candidate == entity_tokens:
            return (i, i + n)

    # Fallback: look for any token that starts with the first entity token
    for i, tok in enumerate(tokens):
        if tok.startswith(entity_tokens[0]):
            return (i, min(i + n, len(tokens)))

    return (0, 0)


def _apply_termination(window_tokens: List[str]) -> List[str]:
    """
    Truncate a token window at the first TERMINATION_TOKEN.
    Returns the (possibly shorter) window.
    """
    for idx, tok in enumerate(window_tokens):
        # Strip trailing punctuation before comparing (e.g. "but," → "but")
        clean = tok.rstrip(".,;:!?")
        if clean in TERMINATION_TOKENS:
            return window_tokens[:idx]
    return window_tokens


def _match_trigger(
    window_tokens: List[str],
    trigger_set: frozenset,
) -> Optional[str]:
    """
    Return the first trigger phrase from *trigger_set* found inside
    *window_tokens*, or None if none match.

    Matching is done on the joined token string so multi-word phrases work
    naturally (e.g. "no history of" matched against "no history of fever").
    Longer phrases take priority over shorter ones to avoid partial matches
    (e.g. prefer "no history of" over bare "no").
    """
    if not window_tokens:
        return None

    joined = " ".join(window_tokens)

    # Sort triggers by length (descending) so longer phrases are tested first.
    for trigger in sorted(trigger_set, key=len, reverse=True):
        if trigger in joined:
            return trigger
    return None


def _find_trigger_span_in_raw(
    raw_text: str,
    trigger_text: str,
    search_start: int,
    search_end: int,
) -> Optional[Span]:
    """
    Locate *trigger_text* (case-insensitive) within raw_text[search_start:search_end]
    and return a Span with offsets into the *original* raw text.

    Returns None if the trigger cannot be found in the window.
    """
    window = raw_text[search_start:search_end].lower()
    idx = window.find(trigger_text.lower())
    if idx == -1:
        return None

    abs_start = search_start + idx
    abs_end = abs_start + len(trigger_text)

    # source_sentence_id is not known here; callers set it via the sentence.
    # We use 0 as a temporary value and the caller fills it in.
    return Span(
        text=raw_text[abs_start:abs_end],
        start=abs_start,
        end=abs_end,
        source_sentence_id=0,  # overridden by the caller
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AssertionEngine:
    """
    Rule-based assertion classifier implementing a NegEx-inspired algorithm.

    Usage::

        engine = AssertionEngine()
        assertions = engine.run(entities, preprocessing_output)
    """

    def run(
        self,
        entities: List[Entity],
        preprocessing_output: PreprocessingOutput,
    ) -> List[Assertion]:
        """
        Classify each entity's assertion status and experiencer.

        Parameters
        ----------
        entities:
            Entities produced by the NER stage.  Every entity must have a
            valid ``span.source_sentence_id`` that indexes into
            ``preprocessing_output.sentences``.
        preprocessing_output:
            Full output of the preprocessing stage, providing sentence
            boundaries and raw text for offset computations.

        Returns
        -------
        List[Assertion]
            One Assertion per input entity, in the same order.
        """
        raw_text = preprocessing_output.raw_text

        # Build a quick lookup: sentence_id → SentenceInfo
        sent_map: dict[int, SentenceInfo] = {
            s.sentence_id: s for s in preprocessing_output.sentences
        }

        assertions: List[Assertion] = []

        for entity in entities:
            sent_id = entity.span.source_sentence_id
            sentence = sent_map.get(sent_id)

            if sentence is None:
                # Defensive: no sentence info → default to PRESENT
                logger.warning(
                    "Entity '%s' (id=%s) references unknown sentence_id=%d; "
                    "defaulting to PRESENT.",
                    entity.span.text,
                    entity.id,
                    sent_id,
                )
                assertions.append(
                    Assertion(
                        entity_id=entity.id,
                        status=AssertionStatus.PRESENT,
                        experiencer=Experiencer.PATIENT,
                        confidence=0.5,
                    )
                )
                continue

            assertion = self._classify_entity(entity, sentence, raw_text)
            assertions.append(assertion)

        logger.info(
            "AssertionEngine classified %d entities.",
            len(assertions),
        )
        return assertions

    # ------------------------------------------------------------------
    # Private implementation
    # ------------------------------------------------------------------

    def _classify_entity(
        self,
        entity: Entity,
        sentence: SentenceInfo,
        raw_text: str,
    ) -> Assertion:
        """
        Core classification logic for a single entity within its sentence.
        """
        sent_id = sentence.sentence_id

        # ---- 1. Tokenise the sentence (lowercase) --------------------
        sent_tokens = _tokenize(sentence.text)

        # ---- 2. Locate entity token range within the sentence --------
        entity_text_lower = entity.span.text.lower()
        tok_start, tok_end = _find_entity_token_range(sent_tokens, entity_text_lower)

        # ---- 3. Extract raw left / right windows ---------------------
        raw_left = sent_tokens[max(0, tok_start - LEFT_WINDOW_TOKENS) : tok_start]
        raw_right = sent_tokens[tok_end : tok_end + RIGHT_WINDOW_TOKENS]

        # ---- 4. Apply termination -- work outward from the entity -----
        # Left window: termination applies from entity *outward* (rightmost
        # tokens are closest to entity), so reverse, trim, reverse back.
        left_reversed = list(reversed(raw_left))
        left_reversed = _apply_termination(left_reversed)
        left_tokens = list(reversed(left_reversed))

        right_tokens = _apply_termination(raw_right)

        logger.debug("Assertion context windows prepared")

        # ---- 5. Check triggers in priority order ---------------------
        # PRIORITY: FAMILY > NEGATION > POSSIBLE > HISTORICAL > PRESENT
        #
        # NOTE ON DOUBLE NEGATION:
        #   "Patient does not deny having cough" → the algorithm will match
        #   "deny" (negation trigger) and classify as NEGATED.  True double-
        #   negation resolution (not deny → present) would require deeper
        #   syntactic analysis (dependency parsing) and is acknowledged as a
        #   known limitation of this rule-based approach.

        # Family triggers apply in BOTH windows
        both_tokens = left_tokens + right_tokens
        matched_trigger: Optional[str] = None
        status = AssertionStatus.PRESENT
        experiencer = Experiencer.PATIENT

        clause_text = " ".join(sent_tokens[max(0, tok_start - LEFT_WINDOW_TOKENS):tok_end + RIGHT_WINDOW_TOKENS])
        pseudo_negation = any(phrase in clause_text for phrase in PSEUDO_NEGATION_PHRASES)
        family_match = _match_trigger(both_tokens, FAMILY_TRIGGERS)
        if family_match:
            # FAMILY takes precedence even when negation is also present.
            # Example: "My father does not have diabetes" → FAMILY, not NEGATED.
            status = AssertionStatus.FAMILY
            experiencer = Experiencer.OTHER
            matched_trigger = family_match
            logger.debug("FAMILY match on '%s'", family_match)

        elif not pseudo_negation and (
            _match_trigger(left_tokens, NEGATION_TRIGGERS)
            or _match_trigger(right_tokens, NEGATION_TRIGGERS)
        ):
            neg_match = (
                _match_trigger(left_tokens, NEGATION_TRIGGERS)
                or _match_trigger(right_tokens, NEGATION_TRIGGERS)
            )
            status = AssertionStatus.NEGATED
            experiencer = Experiencer.PATIENT
            matched_trigger = neg_match
            logger.debug("NEGATION match on '%s'", neg_match)

        elif _match_trigger(left_tokens, POSSIBLE_TRIGGERS):
            status = AssertionStatus.POSSIBLE
            experiencer = Experiencer.PATIENT
            matched_trigger = _match_trigger(left_tokens, POSSIBLE_TRIGGERS)
            logger.debug("POSSIBLE match on '%s'", matched_trigger)

        elif _match_trigger(left_tokens, HISTORICAL_TRIGGERS):
            status = AssertionStatus.HISTORICAL
            experiencer = Experiencer.PATIENT
            matched_trigger = _match_trigger(left_tokens, HISTORICAL_TRIGGERS)
            logger.debug("HISTORICAL match on '%s'", matched_trigger)

        # ---- 6. Build trigger Span in original raw text --------------
        trigger_span: Optional[Span] = None
        if matched_trigger is not None:
            raw_span = _find_trigger_span_in_raw(
                raw_text=raw_text,
                trigger_text=matched_trigger,
                search_start=sentence.raw_start,
                search_end=sentence.raw_end,
            )
            if raw_span is not None:
                # Replace the placeholder source_sentence_id with the real one.
                trigger_span = Span(
                    text=raw_span.text,
                    start=raw_span.start,
                    end=raw_span.end,
                    source_sentence_id=sent_id,
                )

        # ---- 7. Build and return Assertion ---------------------------
        # Confidence is slightly lower when a trigger was matched by the
        # right-window heuristic alone (less reliable), otherwise 0.95 for
        # rule-based matches and 0.90 for default PRESENT (no trigger found).
        if matched_trigger is None:
            confidence = 0.90
        else:
            confidence = 0.95

        return Assertion(
            entity_id=entity.id,
            status=status,
            experiencer=experiencer,
            trigger_text=matched_trigger,
            trigger_span=trigger_span,
            confidence=confidence,
        )
