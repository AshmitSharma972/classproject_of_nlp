"""
Tests for src.preprocessing.preprocessor
=========================================

All tests use only synthetic in-memory data — no file I/O, no network calls.

Design notes
------------
* spaCy and NLTK may not be available in a bare CI environment, so the
  fallback regex splitter is exercised explicitly in
  ``test_sentence_splitting``.
* The ``ClinicalPreprocessor`` is instantiated fresh for each test to keep
  tests fully independent (the class is stateless, so this is cheap).
* ``PreprocessingOutput`` is validated as a proper Pydantic model (not just
  a dict) in ``test_preprocessing_output_schema``.
"""

from __future__ import annotations

import pytest

from src.common.schemas import PreprocessingOutput, SentenceInfo
from src.preprocessing import ClinicalPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def preprocessor() -> ClinicalPreprocessor:
    """Return a fresh, stateless ClinicalPreprocessor instance."""
    return ClinicalPreprocessor()


# ---------------------------------------------------------------------------
# 1. Contraction expansion
# ---------------------------------------------------------------------------

class TestContractionExpansion:
    """Verify that contractions are correctly expanded in normalized_text."""

    def test_ive_had_expands_to_i_have_had(self, preprocessor: ClinicalPreprocessor) -> None:
        """Multi-word contraction 'I've had' → 'i have had'."""
        output = preprocessor.run("I've had fever")
        assert "i have had" in output.normalized_text.lower(), (
            f"Expected 'i have had' in normalized_text; got: {output.normalized_text!r}"
        )

    def test_ive_expands_to_i_have(self, preprocessor: ClinicalPreprocessor) -> None:
        """Single contraction 'I've' → 'i have' when 'had' does not follow."""
        output = preprocessor.run("I've noticed a rash")
        assert "i have" in output.normalized_text.lower()

    def test_dont_expands_to_do_not(self, preprocessor: ClinicalPreprocessor) -> None:
        """\"don't\" → \"do not\"."""
        output = preprocessor.run("I don't have diabetes")
        assert "do not" in output.normalized_text.lower()

    def test_wont_expands_to_will_not(self, preprocessor: ClinicalPreprocessor) -> None:
        """\"won't\" → \"will not\" (irregular form)."""
        output = preprocessor.run("I won't take any medication")
        assert "will not" in output.normalized_text.lower()

    def test_its_expands_to_it_is(self, preprocessor: ClinicalPreprocessor) -> None:
        """\"it's\" → \"it is\"."""
        output = preprocessor.run("It's been three days")
        assert "it is" in output.normalized_text.lower()

    # Parametrized sweep across a selection of contractions.
    @pytest.mark.parametrize(
        "raw, expected_fragment",
        [
            ("I'm feeling weak", "i am"),
            ("I'd take aspirin", "i would"),
            ("I'll return", "i will"),
            ("She's recovering well", "she is"),
            ("They're all healthy", "they are"),
            ("We've been waiting", "we have"),
            ("He's had malaria", "he is"),
            ("couldn't breathe", "could not"),
            ("shouldn't exert", "should not"),
            ("there's no swelling", "there is"),
            ("that's my chief complaint", "that is"),
        ],
    )
    def test_parametrized_contractions(
        self,
        preprocessor: ClinicalPreprocessor,
        raw: str,
        expected_fragment: str,
    ) -> None:
        output = preprocessor.run(raw)
        assert expected_fragment in output.normalized_text.lower(), (
            f"For input {raw!r}: expected {expected_fragment!r} in "
            f"{output.normalized_text!r}"
        )


# ---------------------------------------------------------------------------
# 2. Negation preservation
# ---------------------------------------------------------------------------

class TestNegationPreserved:
    """Negation terms must survive the full preprocessing pipeline intact."""

    def test_not_in_normalized_text(self, preprocessor: ClinicalPreprocessor) -> None:
        """'don't' expands to 'do not'; 'not' must appear in normalized_text."""
        output = preprocessor.run("I don't have diabetes")
        assert "not" in output.normalized_text.lower()

    def test_not_in_tokens(self, preprocessor: ClinicalPreprocessor) -> None:
        """'not' must appear among tokenized tokens in at least one sentence."""
        output = preprocessor.run("I don't have diabetes")
        all_tokens = [t.lower() for si in output.sentences for t in si.tokens]
        assert "not" in all_tokens, f"'not' missing from tokens: {all_tokens}"

    def test_no_preserved(self, preprocessor: ClinicalPreprocessor) -> None:
        """The word 'no' must remain after preprocessing."""
        output = preprocessor.run("There is no swelling and no redness.")
        assert "no" in output.normalized_text.lower()

    def test_denies_preserved(self, preprocessor: ClinicalPreprocessor) -> None:
        """'denies' is a critical clinical negation term — must not be dropped."""
        output = preprocessor.run("Patient denies fever or chills.")
        assert "denies" in output.normalized_text.lower()

    def test_without_preserved(self, preprocessor: ClinicalPreprocessor) -> None:
        """'without' must survive as-is."""
        output = preprocessor.run("Discharged without complications.")
        assert "without" in output.normalized_text.lower()

    def test_absent_preserved(self, preprocessor: ClinicalPreprocessor) -> None:
        """'absent' must survive as-is."""
        output = preprocessor.run("Absent bowel sounds noted.")
        assert "absent" in output.normalized_text.lower()

    def test_never_preserved(self, preprocessor: ClinicalPreprocessor) -> None:
        """'never' must survive as-is."""
        output = preprocessor.run("Patient has never smoked.")
        assert "never" in output.normalized_text.lower()


# ---------------------------------------------------------------------------
# 3. Offset map accuracy
# ---------------------------------------------------------------------------

class TestOffsetMapAccuracy:
    """Verify that normalized-char → raw-char mappings are correct."""

    RAW = "I've had fever for 5 days"

    def test_first_sentence_raw_start_is_zero(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """First sentence's raw_start must be 0 (text begins at char 0)."""
        output = preprocessor.run(self.RAW)
        assert len(output.sentences) >= 1
        assert output.sentences[0].raw_start == 0, (
            f"Expected raw_start=0, got {output.sentences[0].raw_start}"
        )

    def test_offset_map_keys_are_contiguous(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """offset_map must have an entry for every character in normalized_text."""
        output = preprocessor.run(self.RAW)
        for i in range(len(output.normalized_text)):
            assert i in output.offset_map, (
                f"offset_map missing key {i}; normalized_text={output.normalized_text!r}"
            )

    def test_offset_map_values_within_raw_bounds(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """Every offset_map value must be a valid index into raw_text."""
        output = preprocessor.run(self.RAW)
        raw_len = len(output.raw_text)
        for norm_i, raw_i in output.offset_map.items():
            assert 0 <= raw_i < raw_len, (
                f"offset_map[{norm_i}]={raw_i} is out of raw_text bounds (len={raw_len})"
            )

    def test_no_contraction_preserves_identity_map(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """
        For a string with no contractions, the offset_map (before punct
        collapsing) should be mostly identity (modulo lower-casing which
        doesn't change positions).
        """
        raw = "fever for five days"
        output = preprocessor.run(raw)
        # All normalized chars should map to themselves.
        for norm_i, raw_i in output.offset_map.items():
            assert norm_i == raw_i, (
                f"Expected identity map at {norm_i}, got raw_i={raw_i}"
            )

    def test_expansion_shifts_map_correctly(
        self,
        preprocessor: ClinicalPreprocessor,
    ) -> None:
        """
        After expanding "I've" (4 raw chars) → "i have" (6 norm chars), the
        word "fever" that immediately follows the contraction must map back to
        its correct raw position (5 = after "I've ").

        We use "I've fever" rather than "I've had" because "i've had" is a
        recognised multi-word contraction that expands the entire phrase to
        "i have had", mapping all characters to raw pos 0.  "fever" is not
        part of any contraction entry, so its raw offset is unambiguous.

        Raw:        I  '  v  e     f  e  v  e  r
        raw pos:    0  1  2  3  4  5  6  7  8  9
        Normalized: i     h  a  v  e     f  e  v  e  r
        norm pos:   0  1  2  3  4  5  6  7  8  9  10 11
        """
        raw = "I've fever"
        output = preprocessor.run(raw)
        norm_fever_start = output.normalized_text.find("fever")
        assert norm_fever_start != -1, (
            f"'fever' not found in normalized text: {output.normalized_text!r}"
        )
        raw_fever_start = output.offset_map[norm_fever_start]
        # In raw "I've fever", "fever" starts at index 5
        assert raw_fever_start == 5, (
            f"Expected raw offset 5 for 'fever', got {raw_fever_start}; "
            f"normalized={output.normalized_text!r}"
        )


# ---------------------------------------------------------------------------
# 4. Repeated punctuation collapsing
# ---------------------------------------------------------------------------

class TestRepeatedPunctuationCollapsed:
    """3+ identical punctuation marks should be collapsed to one."""

    def test_triple_exclamation_collapsed(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """'fever!!!' → 'fever!' in normalized text."""
        output = preprocessor.run("fever!!!")
        assert "fever!" in output.normalized_text, (
            f"Expected 'fever!' in {output.normalized_text!r}"
        )
        assert "!!!" not in output.normalized_text

    def test_triple_question_collapsed(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """'pain???' → 'pain?'"""
        output = preprocessor.run("pain???")
        assert "pain?" in output.normalized_text
        assert "???" not in output.normalized_text

    def test_ellipsis_collapsed(self, preprocessor: ClinicalPreprocessor) -> None:
        """'...' → '.'"""
        output = preprocessor.run("Pain for 3 days...")
        assert "..." not in output.normalized_text
        # A single period should still be present.
        assert "." in output.normalized_text

    def test_double_punct_collapsed(self, preprocessor: ClinicalPreprocessor) -> None:
        """'..' → '.' (2-char run also collapsed)."""
        output = preprocessor.run("onset..headache")
        assert ".." not in output.normalized_text

    def test_single_punct_unchanged(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """A single punctuation character must not be removed."""
        output = preprocessor.run("No fever. Cough present.")
        assert "." in output.normalized_text

    def test_mixed_punct_not_collapsed(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """Different consecutive punct chars (e.g. '!?') are NOT collapsed."""
        output = preprocessor.run("Really!?")
        # '!' and '?' are different — each appears once, no collapsing.
        assert "!" in output.normalized_text
        assert "?" in output.normalized_text


# ---------------------------------------------------------------------------
# 5. Sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    """Multi-sentence input must yield multiple SentenceInfo objects."""

    def test_two_sentences(self, preprocessor: ClinicalPreprocessor) -> None:
        """Two clearly delimited sentences must produce at least 2 SentenceInfo."""
        output = preprocessor.run("Patient has fever. There is no cough.")
        assert len(output.sentences) >= 2, (
            f"Expected ≥2 sentences, got {len(output.sentences)}: {output.sentences}"
        )

    def test_three_sentences(self, preprocessor: ClinicalPreprocessor) -> None:
        output = preprocessor.run(
            "Chief complaint is chest pain. Patient denies fever. "
            "Shortness of breath is present."
        )
        assert len(output.sentences) >= 3

    def test_sentence_ids_are_zero_based_sequential(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """sentence_id must be 0, 1, 2, … in order."""
        output = preprocessor.run(
            "Fever started three days ago. Cough is mild. No vomiting."
        )
        for expected_id, si in enumerate(output.sentences):
            assert si.sentence_id == expected_id, (
                f"Expected sentence_id={expected_id}, got {si.sentence_id}"
            )

    def test_sentence_raw_offsets_within_raw_text(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """raw_start and raw_end must both fall within the raw_text bounds."""
        raw = "Fever present. Cough absent. No rash."
        output = preprocessor.run(raw)
        raw_len = len(raw)
        for si in output.sentences:
            assert 0 <= si.raw_start < raw_len, (
                f"raw_start={si.raw_start} out of bounds for len={raw_len}"
            )
            assert 0 < si.raw_end <= raw_len, (
                f"raw_end={si.raw_end} out of bounds for len={raw_len}"
            )

    def test_single_sentence_input(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """A single sentence (no terminal punctuation split) returns exactly 1."""
        output = preprocessor.run("Patient reports mild headache")
        assert len(output.sentences) == 1

    def test_sentences_have_tokens(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """Every SentenceInfo must have a non-empty tokens list."""
        output = preprocessor.run("Patient has fever. Cough is present.")
        for si in output.sentences:
            assert len(si.tokens) > 0, (
                f"Sentence {si.sentence_id!r} has no tokens: {si!r}"
            )


# ---------------------------------------------------------------------------
# 6. Spell-flag: known medical term must not be flagged
# ---------------------------------------------------------------------------

class TestSpellFlagNotSilentlyApplied:
    """Verified medical terms must not appear in spell_flags."""

    def test_paracetamol_not_flagged(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """'paracetamol' is in the medical lexicon — must not appear in spell_flags."""
        output = preprocessor.run("Patient is taking paracetamol for the fever.")
        flagged_tokens = {f["token"].lower() for f in output.spell_flags}
        assert "paracetamol" not in flagged_tokens, (
            f"'paracetamol' was incorrectly flagged: {output.spell_flags}"
        )

    def test_ibuprofen_not_flagged(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """'ibuprofen' is in the medical lexicon — must not appear in spell_flags."""
        output = preprocessor.run("Ibuprofen prescribed for arthralgia.")
        flagged_tokens = {f["token"].lower() for f in output.spell_flags}
        assert "ibuprofen" not in flagged_tokens

    def test_spell_flag_structure(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """Every spell flag must have 'token', 'position', and 'suggestion' keys."""
        # Use a near-misspelling to trigger a flag.
        output = preprocessor.run("The patient has fevver and coughing.")
        for flag in output.spell_flags:
            assert "token" in flag, f"Missing 'token' key in flag: {flag}"
            assert "position" in flag, f"Missing 'position' key in flag: {flag}"
            assert "suggestion" in flag, f"Missing 'suggestion' key in flag: {flag}"

    def test_negation_terms_not_flagged(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """Negation terms must never appear in spell_flags."""
        output = preprocessor.run("Patient denies fever and never had diabetes.")
        negation_terms = {"no", "not", "never", "denies", "deny", "denied",
                          "without", "absent"}
        flagged_tokens = {f["token"].lower() for f in output.spell_flags}
        intersection = negation_terms & flagged_tokens
        assert not intersection, (
            f"Negation terms were incorrectly spell-flagged: {intersection}"
        )

    def test_text_without_misspellings_produces_no_flags(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """Clean medical text should produce zero (or very few) spell flags."""
        output = preprocessor.run(
            "Patient has fever. No cough. Paracetamol prescribed."
        )
        # All words are either medical-lexicon or common-English — 0 flags expected.
        assert len(output.spell_flags) == 0, (
            f"Unexpected spell flags: {output.spell_flags}"
        )


# ---------------------------------------------------------------------------
# 7. PreprocessingOutput schema validity
# ---------------------------------------------------------------------------

class TestPreprocessingOutputSchema:
    """The return value must be a valid PreprocessingOutput Pydantic model."""

    def test_is_preprocessing_output_instance(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """run() must return a PreprocessingOutput instance."""
        output = preprocessor.run("Patient has fever and cough.")
        assert isinstance(output, PreprocessingOutput), (
            f"Expected PreprocessingOutput, got {type(output)}"
        )

    def test_raw_text_preserved(self, preprocessor: ClinicalPreprocessor) -> None:
        """raw_text field must equal the original input string."""
        raw = "I've had fever for 5 days."
        output = preprocessor.run(raw)
        assert output.raw_text == raw

    def test_normalized_text_is_string(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """normalized_text must be a non-empty string."""
        output = preprocessor.run("Patient reports pain.")
        assert isinstance(output.normalized_text, str)
        assert len(output.normalized_text) > 0

    def test_sentences_is_list_of_sentence_info(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """sentences must be a list of SentenceInfo objects."""
        output = preprocessor.run("No fever. Mild cough.")
        assert isinstance(output.sentences, list)
        assert all(isinstance(si, SentenceInfo) for si in output.sentences)

    def test_offset_map_is_dict_int_int(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """offset_map must be Dict[int, int]."""
        output = preprocessor.run("Cough for two weeks.")
        assert isinstance(output.offset_map, dict)
        for k, v in output.offset_map.items():
            assert isinstance(k, int) and isinstance(v, int), (
                f"offset_map has non-int entry: {k!r} → {v!r}"
            )

    def test_spell_flags_is_list(self, preprocessor: ClinicalPreprocessor) -> None:
        """spell_flags must be a list."""
        output = preprocessor.run("Patient has fever.")
        assert isinstance(output.spell_flags, list)

    def test_metadata_contains_counts(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """metadata must contain sentence_count and token_count."""
        output = preprocessor.run("Fever for three days. No cough.")
        assert "sentence_count" in output.metadata
        assert "token_count" in output.metadata
        assert isinstance(output.metadata["sentence_count"], int)
        assert isinstance(output.metadata["token_count"], int)

    def test_pydantic_model_serializable(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """The output must be serializable to a dict via .model_dump()."""
        output = preprocessor.run("Patient denies nausea.")
        dumped = output.model_dump()
        assert isinstance(dumped, dict)
        assert "raw_text" in dumped
        assert "normalized_text" in dumped
        assert "sentences" in dumped
        assert "offset_map" in dumped
        assert "spell_flags" in dumped

    def test_empty_input_handled_gracefully(
        self, preprocessor: ClinicalPreprocessor
    ) -> None:
        """An empty string input must not raise an exception."""
        output = preprocessor.run("")
        assert isinstance(output, PreprocessingOutput)
        assert output.raw_text == ""
