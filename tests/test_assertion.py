"""
tests/test_assertion.py

Pytest test suite for the negation / assertion module (src.negation).

All tests use synthetic, in-memory data — no file I/O or network calls.

Covered scenarios
-----------------
* NEGATED  via "No history of" prefix
* NEGATED  via "denies" verb
* POSSIBLE via "maybe" / "possible" hedge
* HISTORICAL via "History of" header
* FAMILY   via "father" family trigger  (experiencer=OTHER)
* PRESENT  default (no trigger)
* PRESENT  for entity after termination token ("but") blocks negation scope
* Assertion.entity_id matches input entity id
* All returned objects are valid Pydantic Assertion models
* Known limitation: double-negation "does not deny" is marked NEGATED
  (documented in assertion_engine.py as a known limitation)

Helper
------
make_entity(text, start, end, sent_id, etype) → Entity
    Builds a minimal Entity with a real UUID for use in tests.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest

from src.common.schemas import (
    Assertion,
    AssertionStatus,
    Entity,
    EntityType,
    Experiencer,
    PreprocessingOutput,
    SentenceInfo,
    Span,
)
from src.negation import AssertionEngine


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def make_entity(
    text: str,
    start: int,
    end: int,
    sent_id: int,
    etype: EntityType = EntityType.DISEASE,
) -> Entity:
    """
    Build a minimal Entity for use in tests.

    Parameters
    ----------
    text:    Verbatim entity text (as it appears in raw_text).
    start:   Character start offset in raw_text (inclusive).
    end:     Character end offset in raw_text (exclusive).
    sent_id: Source sentence index (0-based).
    etype:   EntityType enum value (default: DISEASE).
    """
    return Entity(
        id=str(uuid.uuid4()),
        span=Span(text=text, start=start, end=end, source_sentence_id=sent_id),
        type=etype,
        confidence=1.0,
    )


def make_preprocessing(
    raw_text: str,
    sent_texts: List[str] | None = None,
) -> PreprocessingOutput:
    """
    Build a PreprocessingOutput with a single sentence that spans the full
    raw_text, unless explicit sentence texts are provided.

    Each sentence's tokens are produced by a simple whitespace split.
    """
    if sent_texts is None:
        sent_texts = [raw_text]

    sentences: List[SentenceInfo] = []
    cursor = 0
    for idx, text in enumerate(sent_texts):
        start = raw_text.find(text, cursor)
        if start == -1:
            # Fallback: treat from cursor position
            start = cursor
        end = start + len(text)
        sentences.append(
            SentenceInfo(
                sentence_id=idx,
                text=text,
                raw_start=start,
                raw_end=end,
                tokens=text.split(),
            )
        )
        cursor = end

    return PreprocessingOutput(
        raw_text=raw_text,
        normalized_text=raw_text,
        sentences=sentences,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> AssertionEngine:
    return AssertionEngine()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestNegation:
    """Entities that should be classified as NEGATED."""

    def test_negation_no_history(self, engine: AssertionEngine) -> None:
        """
        'No history of asthma' → the entity 'asthma' should be NEGATED.
        The phrase 'no history of' is a negation trigger in the left window.
        """
        raw = "No history of asthma"
        entity = make_entity("asthma", raw.index("asthma"), raw.index("asthma") + 6, 0)
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert len(results) == 1
        assert results[0].status == AssertionStatus.NEGATED, (
            f"Expected NEGATED, got {results[0].status}. "
            f"Trigger: {results[0].trigger_text!r}"
        )

    def test_negation_denies(self, engine: AssertionEngine) -> None:
        """
        'Patient denies chest pain' → 'chest pain' should be NEGATED.
        'denies' is a negation trigger appearing before the entity.
        """
        raw = "Patient denies chest pain"
        entity = make_entity(
            "chest pain",
            raw.index("chest pain"),
            raw.index("chest pain") + len("chest pain"),
            0,
            EntityType.SYMPTOM,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.NEGATED
        assert results[0].trigger_text == "denies"


class TestPossible:
    """Entities that should be classified as POSSIBLE."""

    def test_possible_maybe(self, engine: AssertionEngine) -> None:
        """'Maybe malaria' → POSSIBLE."""
        raw = "Maybe malaria"
        entity = make_entity(
            "malaria",
            raw.index("malaria"),
            raw.index("malaria") + len("malaria"),
            0,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.POSSIBLE

    def test_possible_keyword(self, engine: AssertionEngine) -> None:
        """'possible malaria' → POSSIBLE."""
        raw = "possible malaria"
        entity = make_entity(
            "malaria",
            raw.index("malaria"),
            raw.index("malaria") + len("malaria"),
            0,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.POSSIBLE


class TestHistorical:
    """Entities that should be classified as HISTORICAL."""

    def test_historical(self, engine: AssertionEngine) -> None:
        """
        'History of hypertension' → HISTORICAL.
        'history of' is a historical trigger in the left window.
        """
        raw = "History of hypertension"
        entity = make_entity(
            "hypertension",
            raw.index("hypertension"),
            raw.index("hypertension") + len("hypertension"),
            0,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.HISTORICAL
        assert results[0].experiencer == Experiencer.PATIENT


class TestFamily:
    """Entities that should be classified as FAMILY with experiencer=OTHER."""

    def test_family_other(self, engine: AssertionEngine) -> None:
        """
        'My father has diabetes' → FAMILY, experiencer=OTHER.
        'father' is a family trigger; family status overrides everything.
        """
        raw = "My father has diabetes"
        entity = make_entity(
            "diabetes",
            raw.index("diabetes"),
            raw.index("diabetes") + len("diabetes"),
            0,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.FAMILY
        assert results[0].experiencer == Experiencer.OTHER

    def test_family_overrides_negation(self, engine: AssertionEngine) -> None:
        """
        'My father does not have diabetes' → FAMILY (not NEGATED).
        FAMILY trigger takes precedence even when negation words are present.
        This tests the documented priority: FAMILY > NEGATION.
        """
        raw = "My father does not have diabetes"
        entity = make_entity(
            "diabetes",
            raw.index("diabetes"),
            raw.index("diabetes") + len("diabetes"),
            0,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.FAMILY, (
            "FAMILY should win over NEGATED even when 'does not have' is present."
        )
        assert results[0].experiencer == Experiencer.OTHER


class TestPresent:
    """Entities that should default to PRESENT."""

    def test_present_default(self, engine: AssertionEngine) -> None:
        """
        'Patient has fever' → PRESENT, experiencer=PATIENT.
        No triggers present; the default assertion is used.
        """
        raw = "Patient has fever"
        entity = make_entity(
            "fever",
            raw.index("fever"),
            raw.index("fever") + len("fever"),
            0,
            EntityType.SYMPTOM,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.PRESENT
        assert results[0].experiencer == Experiencer.PATIENT


class TestTermination:
    """Tests that TERMINATION_TOKENS properly limit trigger scope."""

    def test_termination_token_stops_scope(self, engine: AssertionEngine) -> None:
        """
        'No fever, but he has chest pain'
        → 'chest pain' should be PRESENT.

        'No' is a negation trigger but 'but' terminates the negation scope
        before 'chest pain', so 'chest pain' is not negated.
        """
        raw = "No fever, but he has chest pain"
        # 'chest pain' appears after 'but'
        cp_start = raw.index("chest pain")
        entity = make_entity(
            "chest pain",
            cp_start,
            cp_start + len("chest pain"),
            0,
            EntityType.SYMPTOM,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].status == AssertionStatus.PRESENT, (
            f"'chest pain' after 'but' should be PRESENT, got {results[0].status}. "
            "The termination token 'but' should stop the negation scope."
        )


class TestSchemaAndLinkage:
    """Tests for schema validity and entity-id linkage."""

    def test_assertion_linked_to_entity(self, engine: AssertionEngine) -> None:
        """Assertion.entity_id must match the input entity's id exactly."""
        raw = "Patient has headache"
        entity = make_entity(
            "headache",
            raw.index("headache"),
            raw.index("headache") + len("headache"),
            0,
            EntityType.SYMPTOM,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        assert results[0].entity_id == entity.id

    def test_assertion_schema_valid(self, engine: AssertionEngine) -> None:
        """All returned objects must be valid Pydantic Assertion instances."""
        raw = "No history of asthma. Patient has fever. My father has diabetes."
        entities = [
            make_entity("asthma", raw.index("asthma"), raw.index("asthma") + 6, 0),
            make_entity("fever", raw.index("fever"), raw.index("fever") + 5, 1, EntityType.SYMPTOM),
            make_entity("diabetes", raw.index("diabetes"), raw.index("diabetes") + 8, 2),
        ]
        # Three sentences for the three statements
        pp = make_preprocessing(
            raw,
            sent_texts=[
                "No history of asthma.",
                "Patient has fever.",
                "My father has diabetes.",
            ],
        )

        results = engine.run(entities, pp)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, Assertion), (
                f"Expected Assertion instance, got {type(result)}"
            )
            # Pydantic model_validate would raise if the object were invalid,
            # but we also check the key fields are sensible types.
            assert isinstance(result.entity_id, str)
            assert isinstance(result.status, AssertionStatus)
            assert isinstance(result.experiencer, Experiencer)
            assert 0.0 <= result.confidence <= 1.0

    def test_one_assertion_per_entity(self, engine: AssertionEngine) -> None:
        """engine.run must return exactly one Assertion per input Entity."""
        raw = "Patient has fever and cough"
        entities = [
            make_entity("fever", raw.index("fever"), raw.index("fever") + 5, 0, EntityType.SYMPTOM),
            make_entity("cough", raw.index("cough"), raw.index("cough") + 5, 0, EntityType.SYMPTOM),
        ]
        pp = make_preprocessing(raw)

        results = engine.run(entities, pp)

        assert len(results) == len(entities)


class TestKnownLimitations:
    """
    Tests that document known limitations of the rule-based approach.
    These tests verify the current (imperfect) behaviour rather than the
    theoretically correct one, so future improvements are trackable.
    """

    def test_double_negation_not_inverted(self, engine: AssertionEngine) -> None:
        """
        'Patient does not deny having cough' contains 'deny' (a negation trigger).
        A true double-negation resolver would classify this as PRESENT, but our
        rule-based engine lacks the syntactic depth for that.

        KNOWN LIMITATION: This test documents that the engine will classify
        the entity as NEGATED.  If a future version adds double-negation
        resolution, update this test accordingly.
        """
        raw = "Patient does not deny having cough"
        entity = make_entity(
            "cough",
            raw.index("cough"),
            raw.index("cough") + len("cough"),
            0,
            EntityType.SYMPTOM,
        )
        pp = make_preprocessing(raw)

        results = engine.run([entity], pp)

        # Document the known limitation: currently NEGATED, not PRESENT.
        # This is acceptable and explicitly noted in assertion_engine.py.
        assert results[0].status == AssertionStatus.NEGATED, (
            "KNOWN LIMITATION: double-negation 'does not deny' is classified as "
            "NEGATED by the current rule-based engine. This is intentional — see "
            "assertion_engine.py for the documented limitation."
        )
