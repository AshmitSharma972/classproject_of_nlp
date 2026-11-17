"""
triggers.py -- Trigger word/phrase sets and scanning constants for the
NegEx-inspired assertion / negation module.

All sets use lowercase strings so callers must lower-case the windowed text
before matching.  Multi-word phrases (e.g. 'no history of') are matched as
joined token sequences, so they must appear as single entries here.
"""

# ---------------------------------------------------------------------------
# Negation triggers
# Phrases that assert an entity is ABSENT or DENIED for the patient.
# ---------------------------------------------------------------------------
NEGATION_TRIGGERS: frozenset = frozenset({
    # Single-word negators
    "no",
    "not",
    "never",
    "without",
    "absent",
    "neither",
    "nor",
    "cannot",
    # Verb-based negators
    "denies",
    "deny",
    "denied",
    # Adjective / prepositional phrases
    "negative for",
    "free of",
    "clear of",
    # Multi-word clinical negators
    "rules out",
    "ruled out",
    "unremarkable for",
    "no evidence of",
    "no sign of",
    "no history of",
    "not had",
    # Contraction forms (tokenised and rejoined)
    "could not",
    "doesn't have",
    "does not have",
    "cannot have",
    "can not",
})

# ---------------------------------------------------------------------------
# Possibility / uncertainty triggers
# Phrases that mark an entity as uncertain or speculative.
# ---------------------------------------------------------------------------
POSSIBLE_TRIGGERS: frozenset = frozenset({
    # Single-word hedges
    "possible",
    "possibly",
    "probable",
    "probably",
    "suspect",
    "suspected",
    "questionable",
    "uncertain",
    "not certain",
    "unclear",
    "perhaps",
    "maybe",
    # Multi-word hedges
    "may have",
    "might have",
    "cannot rule out",
    "cannot exclude",
    "rule out",         # pre-diagnosis "rule out X" → uncertain status
})

# ---------------------------------------------------------------------------
# Historical triggers
# Phrases that place an entity in the patient's past rather than current state.
# ---------------------------------------------------------------------------
HISTORICAL_TRIGGERS: frozenset = frozenset({
    # Common history headers
    "history of",
    "past history",
    "past medical history",
    "pmh",
    # Temporal distance markers
    "prior to",
    "previously",
    "years ago",
    "months ago",
    "weeks ago",
    "days ago",
    # Clinical shorthand
    "former",
    "known case of",
    "treated for",
    "status post",
    "s/p",
    "old",
    "remote",
    # Narrative phrases
    "previous episode of",
    "in the past",
    "prior history",
})

# ---------------------------------------------------------------------------
# Family / other-experiencer triggers
# Phrases that indicate the clinical finding belongs to a family member,
# not the patient themselves.
# ---------------------------------------------------------------------------
FAMILY_TRIGGERS: frozenset = frozenset({
    # Immediate family
    "father",
    "mother",
    "brother",
    "sister",
    "son",
    "daughter",
    # Extended family
    "grandfather",
    "grandmother",
    "uncle",
    "aunt",
    "cousin",
    # Generic terms
    "parent",
    "sibling",
    "husband",
    "wife",
    # Clinical header phrases
    "family history of",
    "family h/o",
    "familial",
    "hereditary",
    "runs in the family",
    # Directional qualifiers
    "maternal",
    "paternal",
})

# ---------------------------------------------------------------------------
# Scanning window sizes (in tokens)
# ---------------------------------------------------------------------------

# How many tokens to the LEFT of an entity to inspect for triggers.
LEFT_WINDOW_TOKENS: int = 6

# How many tokens to the RIGHT of an entity to inspect for triggers.
RIGHT_WINDOW_TOKENS: int = 4

# ---------------------------------------------------------------------------
# Termination tokens
# If one of these tokens appears in the window, scanning stops at that point.
# This prevents triggers in one clause from bleeding into adjacent clauses.
# ---------------------------------------------------------------------------
TERMINATION_TOKENS: frozenset = frozenset({
    "but",
    "however",
    "although",
    "except",
    "nevertheless",
    "yet",
    "still",
    "and",
    ";",
})

PSEUDO_NEGATION_PHRASES: frozenset = frozenset({
    "no change in", "cannot deny", "can not deny",
    "not only", "not certain",
})
