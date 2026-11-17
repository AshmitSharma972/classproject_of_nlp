"""
Clinical Text Preprocessor
==========================

Transforms raw free-text patient narrations into a normalized, structured
:class:`~src.common.schemas.PreprocessingOutput` ready for downstream NLP
stages.

Key responsibilities
--------------------
1. **Contraction expansion** - expands English contractions *with full
   character-level offset tracking* so every downstream span can be mapped
   back to the original raw text.
2. **Punctuation collapsing** - collapses 3 + identical punctuation marks
   (e.g. ``!!!`` → ``!``), preserving clinical meaning.
3. **Negation guard** - the set of clinical negation tokens is *never*
   silently removed or altered.
4. **Medical spell-guard** - flags (never fixes) tokens that look like
   possible misspellings via simple edit-distance heuristics against a
   built-in medical lexicon and a small general English word list.
5. **Sentence splitting** - tries spaCy ``en_core_web_sm`` first, then
   NLTK ``sent_tokenize``, then a regex fallback.
6. **Tokenization** - regex ``\\b[\\w/'-]+\\b`` preserving hyphenated /
   slash terms.

All character offsets in every output object reference the **original
raw input text**, not the normalized text.
"""

from __future__ import annotations

import re
import hashlib
from typing import Dict, List, Optional, Tuple

from src.common.schemas import PreprocessingOutput, SentenceInfo
from src.utils import get_logger

logger = get_logger("ClinicalPreprocessor")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Longest-match-first: sort by descending key length so multi-word
# contractions (e.g. "i've had") are matched before shorter ones.
_RAW_CONTRACTIONS: Dict[str, str] = {
    "i've had":   "i have had",
    "i've":       "i have",
    "i'm":        "i am",
    "i'd":        "i would",
    "i'll":       "i will",
    "don't":      "do not",
    "doesn't":    "does not",
    "didn't":     "did not",
    "can't":      "cannot",
    "couldn't":   "could not",
    "won't":      "will not",
    "wouldn't":   "would not",
    "isn't":      "is not",
    "aren't":     "are not",
    "wasn't":     "was not",
    "weren't":    "were not",
    "hasn't":     "has not",
    "haven't":    "have not",
    "hadn't":     "had not",
    "shouldn't":  "should not",
    "mustn't":    "must not",
    "needn't":    "need not",
    "there's":    "there is",
    "that's":     "that is",
    "it's":       "it is",
    "he's":       "he is",
    "she's":      "she is",
    "they're":    "they are",
    "we're":      "we are",
    "you're":     "you are",
    "they've":    "they have",
    "we've":      "we have",
    "you've":     "you have",
    "they'd":     "they would",
    "we'd":       "we would",
    "you'd":      "you would",
}

# Sort descending by key length for longest-match-first greedy replacement.
CONTRACTIONS: List[Tuple[str, str]] = sorted(
    _RAW_CONTRACTIONS.items(), key=lambda kv: len(kv[0]), reverse=True
)

# Compiled regex patterns for each contraction (word-boundary aware).
# We compile once at module load time for efficiency.
_CONTRACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)", re.IGNORECASE), exp)
    for c, exp in CONTRACTIONS
]

# Negation terms that must NEVER be silently removed or altered.
NEGATION_TERMS: frozenset = frozenset(
    ["no", "not", "never", "denies", "deny", "denied", "without", "absent"]
)

# Regex that collapses 3+ consecutive identical punctuation characters
# to a single one (e.g. "!!!" → "!", "..." → ".").
# We also collapse exactly 2 dots ("..") to "." for clinical safety,
# but leave single characters untouched.
_PUNCT_COLLAPSE_RE = re.compile(r"([^\w\s])\1{1,}")

# Token pattern: word chars, forward-slashes, hyphens, apostrophes.
_TOKEN_RE = re.compile(r"\b[\w/'-]+\b")

# ---------------------------------------------------------------------------
# Medical lexicon (built-in, ~100 common terms)
# ---------------------------------------------------------------------------

MEDICAL_LEXICON: frozenset = frozenset(
    [
        # Symptoms & general
        "fever", "cough", "dyspnea", "tachycardia", "bradycardia",
        "hypertension", "hypotension", "nausea", "vomiting", "diarrhea",
        "constipation", "fatigue", "malaise", "syncope", "palpitations",
        "edema", "anorexia", "polydipsia", "polyuria", "oliguria",
        "hemoptysis", "hematemesis", "melena", "hematuria", "jaundice",
        "pruritus", "erythema", "cyanosis", "pallor", "diaphoresis",
        "arthralgia", "myalgia", "headache", "vertigo", "tinnitus",
        "diplopia", "dysphagia", "odynophagia", "epistaxis", "rhinorrhea",
        # Diagnoses
        "diabetes", "hypertension", "asthma", "pneumonia", "tuberculosis",
        "malaria", "dengue", "typhoid", "hepatitis", "cirrhosis",
        "appendicitis", "cholecystitis", "pancreatitis", "gastritis",
        "gastroenteritis", "urticaria", "eczema", "psoriasis",
        "hypothyroidism", "hyperthyroidism", "anemia", "leukemia",
        "lymphoma", "sepsis", "meningitis", "encephalitis", "migraine",
        "epilepsy", "ischemia", "infarction", "atherosclerosis",
        # Medications
        "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "amoxicillin",
        "azithromycin", "ciprofloxacin", "metformin", "insulin", "atorvastatin",
        "amlodipine", "metoprolol", "lisinopril", "warfarin", "heparin",
        "clopidogrel", "omeprazole", "pantoprazole", "ranitidine", "cetirizine",
        "loratadine", "salbutamol", "prednisolone", "dexamethasone",
        "hydrochlorothiazide", "furosemide", "spironolactone",
        # Body parts / anatomy
        "abdomen", "thorax", "pelvis", "cranium", "sternum", "femur",
        "tibia", "fibula", "humerus", "radius", "ulna", "patella",
        "clavicle", "scapula", "vertebra", "lumbar", "cervical", "sacral",
        "hepatic", "splenic", "renal", "pulmonary", "cardiac", "cerebral",
        # Tests / investigations
        "ecg", "ekg", "mri", "ctscan", "ultrasound", "xray", "biopsy",
        "endoscopy", "colonoscopy", "spirometry", "echocardiography",
        "hemoglobin", "hematocrit", "creatinine", "bilirubin", "albumin",
        "troponin", "ferritin", "procalcitonin",
    ]
)

# Minimal English stopword / common-word set used to skip spell-flagging
# ordinary English words that are fine but not in the medical lexicon.
_COMMON_ENGLISH: frozenset = frozenset(
    [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "may", "might", "can", "could",
        "must", "need", "dare", "ought", "used", "that", "this", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "not", "no", "never", "none", "nor", "so", "yet", "both", "either",
        "neither", "each", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "because", "as", "until", "while",
        "although", "though", "if", "when", "where", "who", "which", "what",
        "how", "all", "any", "any", "both", "day", "days", "week", "weeks",
        "month", "months", "year", "years", "ago", "since", "during", "after",
        "before", "patient", "complaint", "history", "present", "past",
        "family", "social", "drug", "allergy", "review", "system",
        "has", "have", "had", "been", "also", "well", "old", "new",
        "denies", "deny", "denied", "without", "absent", "onset",
        "reports", "report", "reported", "pain", "chest", "back",
        "leg", "arm", "head", "neck", "eye", "ear", "nose", "throat",
        "skin", "heart", "lung", "liver", "kidney", "stomach",
        "blood", "urine", "stool", "weight", "loss", "gain",
        "mild", "moderate", "severe", "chronic", "acute", "bilateral",
        "left", "right", "upper", "lower", "per", "mg", "ml", "mcg",
        "bp", "hr", "rr", "spo2", "temp",
    ]
)


# ---------------------------------------------------------------------------
# Helper: edit distance (Levenshtein) -- simple O(n*m) DP for small strings
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Use two-row DP for memory efficiency.
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            )
        prev = curr
    return prev[lb]


def _find_closest_medical_term(token: str) -> Optional[str]:
    """
    Return the closest medical lexicon term whose edit distance from *token*
    is ≤ 2 (and < half the token length to avoid wildly wrong suggestions).
    Returns None if no sufficiently close term is found.
    """
    best_term: Optional[str] = None
    best_dist: int = 3  # threshold exclusive upper bound

    for term in MEDICAL_LEXICON:
        # Quick length filter: skip if length difference > 3.
        if abs(len(term) - len(token)) > 3:
            continue
        dist = _levenshtein(token, term)
        if dist < best_dist and dist < len(token) // 2 + 1:
            best_dist = dist
            best_term = term

    return best_term


# ---------------------------------------------------------------------------
# Sentence splitter helpers
# ---------------------------------------------------------------------------

def _split_sentences_spacy(text: str) -> List[str]:
    """Try spaCy en_core_web_sm sentence segmentation."""
    import spacy  # type: ignore
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _split_sentences_nltk(text: str) -> List[str]:
    """Fallback to NLTK sent_tokenize."""
    from nltk import sent_tokenize  # type: ignore
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def _split_sentences_regex(text: str) -> List[str]:
    """Last-resort regex sentence splitter on . ! ? followed by whitespace."""
    protected = re.sub(r"\b(?:dr|mr|mrs|ms)\.", lambda m: m.group(0).replace(".", "<DOT>"), text)
    protected = re.sub(r"\b(?:b\.i\.d|t\.i\.d|q\.i\.d|o\.d)\.", lambda m: m.group(0).replace(".", "<DOT>"), protected, flags=re.I)
    protected = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", "<DOT>"), protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    parts = [part.replace("<DOT>", ".") for part in parts]
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> List[str]:
    """
    Attempt sentence splitting with the spaCy → NLTK → regex fallback chain.
    Always returns at least one sentence (the full text) if all splitters fail.
    """
    protected = re.sub(r"\b(?:dr|mr|mrs|ms)\.", lambda m: m.group(0).replace(".", "<CLINIQDOT>"), text, flags=re.I)
    protected = re.sub(r"\b(?:b\.i\.d|t\.i\.d|q\.i\.d|o\.d)\.", lambda m: m.group(0).replace(".", "<CLINIQDOT>"), protected, flags=re.I)
    for splitter_name, splitter_fn in [
        ("spaCy", _split_sentences_spacy),
        ("NLTK", _split_sentences_nltk),
        ("regex", _split_sentences_regex),
    ]:
        try:
            sentences = splitter_fn(protected)
            if sentences:
                logger.debug("Sentence splitting succeeded with %s", splitter_name)
                return [s.replace("<CLINIQDOT>", ".") for s in sentences]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Sentence splitter '%s' failed (%s). Trying next fallback.",
                splitter_name,
                exc,
            )
    # Ultimate fallback: treat everything as a single sentence.
    logger.warning("All sentence splitters failed; treating entire text as one sentence.")
    return [text]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClinicalPreprocessor:
    """
    Stateless clinical text pre-processor.

    Instantiate once and call :meth:`run` for each document.

    Example
    -------
    ::

        preprocessor = ClinicalPreprocessor()
        output = preprocessor.run("I've had fever for 5 days.")
        print(output.normalized_text)
        # → "i have had fever for 5 days."
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, raw_text: str) -> PreprocessingOutput:
        """
        Normalize *raw_text* and return a fully-populated
        :class:`~src.common.schemas.PreprocessingOutput`.

        Parameters
        ----------
        raw_text:
            The verbatim patient narration string.

        Returns
        -------
        PreprocessingOutput
            Normalized text, sentence list with raw offsets, offset map,
            and spell flags.
        """
        logger.info("Starting preprocessing (raw length=%d chars).", len(raw_text))

        # Step 1: lower-case copy for matching (offsets still track raw).
        lowered = raw_text.lower()

        # Step 2: expand contractions and build char-level offset map.
        normalized, offset_map = self._expand_contractions_with_offsets(lowered, raw_text)

        # Step 3: collapse repeated punctuation.
        normalized, offset_map = self._collapse_punctuation(normalized, offset_map)

        # Step 4: sentence splitting on normalized text.
        raw_sentences = _split_sentences(normalized)

        # Step 5: for each sentence, compute raw offsets and tokenize.
        sentence_infos: List[SentenceInfo] = []
        search_cursor = 0  # position in normalized_text to avoid false matches

        for s_id, sent_text in enumerate(raw_sentences):
            # Locate this sentence's start in the normalized text.
            norm_start = normalized.find(sent_text, search_cursor)
            if norm_start == -1:
                # Shouldn't happen, but guard against edge cases.
                norm_start = search_cursor
            norm_end = norm_start + len(sent_text)
            search_cursor = norm_end

            # Map normalized offsets → raw offsets via offset_map.
            raw_start = offset_map.get(norm_start, norm_start)
            # For the end, use the last character of the sentence.
            raw_end_char_norm = norm_end - 1
            raw_end_char_raw = offset_map.get(raw_end_char_norm, raw_end_char_norm)
            # raw_end is exclusive, so add 1.
            raw_end = raw_end_char_raw + 1

            tokens = _TOKEN_RE.findall(sent_text)

            sentence_infos.append(
                SentenceInfo(
                    sentence_id=s_id,
                    text=sent_text,
                    raw_start=raw_start,
                    raw_end=raw_end,
                    tokens=tokens,
                )
            )

        # Step 6: spell flags.
        all_tokens: List[str] = _TOKEN_RE.findall(normalized)
        spell_flags = self._build_spell_flags(all_tokens, normalized)

        # Step 7: compute aggregate stats.
        total_tokens = sum(len(si.tokens) for si in sentence_infos)

        text_digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
        logger.info(
            "Preprocessing complete: %d sentence(s), %d token(s), raw_length=%d, raw_sha256=%s",
            len(sentence_infos),
            total_tokens,
            len(raw_text),
            text_digest,
        )

        return PreprocessingOutput(
            raw_text=raw_text,
            normalized_text=normalized,
            sentences=sentence_infos,
            offset_map=offset_map,
            spell_flags=spell_flags,
            metadata={
                "sentence_count": len(sentence_infos),
                "token_count": total_tokens,
                "contraction_map_size": len(CONTRACTIONS),
                "spell_flag_count": len(spell_flags),
            },
        )

    # ------------------------------------------------------------------
    # Step 1 - Contraction expansion with offset tracking
    # ------------------------------------------------------------------

    def _expand_contractions_with_offsets(
        self,
        lowered: str,
        raw_text: str,
    ) -> Tuple[str, Dict[int, int]]:
        """
        Expand contractions in *lowered* while building a character-level
        ``offset_map``: ``{norm_pos: raw_pos}`` for every character in the
        resulting normalized string.

        The algorithm walks through the lowered text left-to-right.  At each
        position it checks all contraction patterns (longest first).  When a
        match is found at the *current* position, the contraction is replaced
        by its expansion and every character of the expansion maps back to the
        start raw offset of the original contraction (conservative but
        correct: the expanded characters all "come from" the contraction span).
        Non-matched characters map 1-to-1 to their raw positions.

        Parameters
        ----------
        lowered:
            Lower-cased version of the original text (same length as raw_text).
        raw_text:
            The original raw text (used only for provenance; not mutated).

        Returns
        -------
        (normalized_text, offset_map)
        """
        # We build a list of (norm_char, raw_pos) tuples, then join.
        result_chars: List[str] = []
        result_raw_positions: List[int] = []

        raw_pos = 0  # current position in the raw / lowered string
        text_len = len(lowered)

        while raw_pos < text_len:
            # Zero-width joiners are formatting characters.  Removing them
            # improves Indic token matching without altering source offsets.
            if lowered[raw_pos] in ("\u200c", "\u200d"):
                raw_pos += 1
                continue
            matched = False

            # Try each contraction pattern (already sorted longest-first).
            for pattern, expansion in _CONTRACTION_PATTERNS:
                m = pattern.match(lowered, raw_pos)
                if m and m.start() == raw_pos:
                    # Found a contraction starting exactly at raw_pos.
                    contraction_raw_start = raw_pos
                    contraction_raw_end = m.end()  # exclusive

                    # Append each character of the expansion; all map back
                    # to the start of the original contraction span.
                    for exp_char in expansion:
                        result_chars.append(exp_char)
                        result_raw_positions.append(contraction_raw_start)

                    # Advance past the entire original contraction.
                    raw_pos = contraction_raw_end
                    matched = True
                    break  # longest-match already ensured by sort order

            if not matched:
                # Copy character verbatim; 1-to-1 offset.
                result_chars.append(lowered[raw_pos])
                result_raw_positions.append(raw_pos)
                raw_pos += 1

        normalized = "".join(result_chars)
        offset_map: Dict[int, int] = {
            norm_i: raw_i
            for norm_i, raw_i in enumerate(result_raw_positions)
        }
        return normalized, offset_map

    # ------------------------------------------------------------------
    # Step 2 - Punctuation collapsing with offset-map update
    # ------------------------------------------------------------------

    def _collapse_punctuation(
        self,
        text: str,
        offset_map: Dict[int, int],
    ) -> Tuple[str, Dict[int, int]]:
        """
        Collapse 2+ consecutive identical punctuation characters to a single
        one (e.g. ``!!!`` → ``!``, ``...`` → ``!``).

        The offset map is recomputed so the collapsed text's positions still
        point to the correct raw positions.

        Negation tokens are NOT affected because they contain no punctuation.
        """
        result_chars: List[str] = []
        result_raw_positions: List[int] = []

        i = 0
        while i < len(text):
            ch = text[i]
            # Check if this is a punctuation char that starts a run of 2+.
            if not ch.isalnum() and not ch.isspace():
                # Count run length.
                j = i + 1
                while j < len(text) and text[j] == ch:
                    j += 1
                run_len = j - i

                # Emit exactly one character regardless of run length.
                result_chars.append(ch)
                # The emitted char maps to the raw position of the first char
                # in the run (conservative: keep earliest provenance).
                result_raw_positions.append(offset_map.get(i, i))

                i = j  # skip the rest of the run
            else:
                result_chars.append(ch)
                result_raw_positions.append(offset_map.get(i, i))
                i += 1

        new_text = "".join(result_chars)
        new_offset_map: Dict[int, int] = {
            norm_i: raw_i
            for norm_i, raw_i in enumerate(result_raw_positions)
        }
        return new_text, new_offset_map

    # ------------------------------------------------------------------
    # Step 3 - Spell flagging (flag, never fix)
    # ------------------------------------------------------------------

    def _build_spell_flags(
        self,
        tokens: List[str],
        normalized_text: str,
    ) -> List[Dict]:
        """
        Scan *tokens* and flag any that:

        - Are not in the medical lexicon, AND
        - Are not in the common English word list, AND
        - Are longer than 3 characters (avoid flagging abbreviations like
          "BP", "HR"), AND
        - Are not pure digits or symbols, AND
        - Have a close neighbour in the medical lexicon (edit distance ≤ 2),
          which suggests an actual misspelling rather than a proper noun or
          unknown term.

        Negation terms are exempt from flagging.

        Each flag is a dict with keys ``token``, ``position`` (character
        offset in *normalized_text*), and ``suggestion`` (closest medical
        term or ``None``).
        """
        flags: List[Dict] = []
        seen: set = set()  # avoid duplicate flags for repeated tokens

        # Build a position index: token → first occurrence in normalized_text.
        # We search left-to-right to get approximate positions.
        search_cursor = 0

        for tok in tokens:
            tok_lower = tok.lower()

            # Skip if already flagged.
            if tok_lower in seen:
                continue

            # Skip negation terms (must never be modified).
            if tok_lower in NEGATION_TERMS:
                continue

            # Skip known medical terms.
            if tok_lower in MEDICAL_LEXICON:
                continue

            # Skip common English words.
            if tok_lower in _COMMON_ENGLISH:
                continue

            # Skip very short tokens (≤ 3 chars) -- likely abbreviations.
            if len(tok_lower) <= 3:
                continue

            # Skip tokens that are purely numeric or alphanumeric short codes.
            if tok_lower.isdigit():
                continue

            # Find position of this token in normalized_text.
            tok_pos = normalized_text.find(tok, search_cursor)
            if tok_pos == -1:
                tok_pos = normalized_text.find(tok)  # fallback: search from start
            if tok_pos != -1:
                search_cursor = tok_pos + len(tok)

            # Look for a close medical-lexicon neighbour.
            suggestion = _find_closest_medical_term(tok_lower)

            # Only flag if we found a plausible medical suggestion
            # (otherwise it might just be a valid word we don't know).
            if suggestion is not None:
                flags.append(
                    {
                        "token": tok,
                        "position": tok_pos if tok_pos != -1 else -1,
                        "suggestion": suggestion,
                    }
                )
                seen.add(tok_lower)
                logger.debug(
                    "Spell flag: %r at pos %d -- suggestion: %r",
                    tok, tok_pos, suggestion,
                )

        return flags
