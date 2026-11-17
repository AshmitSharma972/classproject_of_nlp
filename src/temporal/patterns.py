"""
Temporal expression patterns for the clinical NLP pipeline.

All regex patterns are designed for case-insensitive matching.
Patterns are compiled once and reused across all processing runs.
"""
import re
from typing import List

# ---------------------------------------------------------------------------
# Number word → digit mapping (for normalization)
# ---------------------------------------------------------------------------
NUMBER_WORDS: dict = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}

# ---------------------------------------------------------------------------
# Duration patterns  (e.g.  "5 days", "two weeks", "for the past 3 months")
# ---------------------------------------------------------------------------
_DURATION_RAW: List[str] = [
    r"\bsince\s+(childhood|birth)\b",
    r"\bfor\s+(?:the\s+(?:past|last)\s+)?(\w+)\s+(days?|weeks?|months?|years?|hours?|minutes?)\b",
    r"\bsince\s+(\w+)\s+(days?|weeks?|months?|years?|hours?)\b",
    r"\b(\w+)\s+(days?|weeks?|months?|years?|hours?)\s+ago\b",
    r"\b(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|years?|hours?|minutes?)\s+(?:ago|back|before|prior)\b",
    r"\b(\d+(?:\.\d+)?)\s*-\s*(days?|weeks?|months?|years?|hours?)\b",
    r"\b(several|few|many|couple\s+of)\s+(days?|weeks?|months?|years?|hours?)\b",
    r"\bfor\s+(\d+(?:\.\d+)?)\s*(days?|weeks?|months?|years?|hours?)\b",
]
DURATION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _DURATION_RAW
]

# Standalone simple durations (matched first for the full phrase)
SIMPLE_DURATION_PATTERN: re.Pattern = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|"
    r"forty|fifty|several|few|many)\s+(days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Onset patterns  (e.g.  "started 3 days ago", "onset 2 weeks back")
# ---------------------------------------------------------------------------
_ONSET_RAW: List[str] = [
    r"\b(?:started|began|started\s+having|developed|noticed|onset\s+of?)\s+(?:(?:the\s+)?(?:\w+\s+){0,3})?(\d+|"
    r"one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?|hours?)\s+ago\b",
    r"\b(?:sudden|gradual|acute|insidious)\s+onset\b",
    r"\bonset\s+(?:was|is)?\s*(\d+)\s*(days?|weeks?|months?)\s+ago\b",
]
ONSET_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _ONSET_RAW
]

# ---------------------------------------------------------------------------
# Relative date patterns (e.g. "last week", "yesterday", "this morning")
# ---------------------------------------------------------------------------
_RELATIVE_RAW: List[str] = [
    r"\b(yesterday|today|this\s+morning|last\s+night|the\s+night\s+before)\b",
    r"\blast\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bthis\s+(week|month|year|morning|afternoon|evening)\b",
    r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+(?:ago|back|before)\b",
    r"\brecently\b",
    r"\bsince\s+(childhood|birth)\b",
    r"\ba\s+(few|couple\s+of)\s+(days?|weeks?|months?)\s+ago\b",
]
RELATIVE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _RELATIVE_RAW
]

# ---------------------------------------------------------------------------
# Absolute date patterns  (e.g.  "in 2019", "on 12/05/2023", "March 2022")
# ---------------------------------------------------------------------------
_DATE_RAW: List[str] = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b",
    r"\bin\s+(20\d{2}|19\d{2})\b",
    r"\b(?:early|mid|late)\s+(?:20\d{2}|19\d{2})\b",
]
DATE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _DATE_RAW
]

# ---------------------------------------------------------------------------
# Previous episode / anaphora patterns
# ---------------------------------------------------------------------------
_EPISODE_RAW: List[str] = [
    r"\bsimilar\s+(?:problem|episode|illness|complaint|symptoms?)\b",
    r"\bprevious\s+(?:episode|attack|bout|occurrence)\b",
    r"\blast\s+time\b",
    r"\bsame\s+(?:problem|complaint|symptoms?|condition)\b",
    r"\b(?:recurrence|recurrent|relapse|relapsed)\b",
]
EPISODE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _EPISODE_RAW
]

# ---------------------------------------------------------------------------
# Anaphora resolution cue patterns  (the X → links to earlier X)
# ---------------------------------------------------------------------------
ANAPHORA_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bthe\s+(fever|cough|pain|rash|swelling|infection|disease|condition|illness|episode)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+(fever|cough|pain|rash|swelling|infection|disease|condition|illness)\b", re.IGNORECASE),
    re.compile(r"\bsame\s+(symptoms?|complaints?)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
UNIT_NORMALIZATION: dict = {
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks",
    "month": "months", "months": "months",
    "year": "years", "years": "years",
    "hour": "hours", "hours": "hours",
    "minute": "minutes", "minutes": "minutes",
}


def normalize_duration(quantity_str: str, unit_str: str) -> str:
    """Return a normalized duration string like '5 days', '2 weeks'."""
    qty_lower = quantity_str.lower().strip()
    qty = NUMBER_WORDS.get(qty_lower, None)
    if qty is None:
        try:
            qty = float(qty_lower)
            qty = int(qty) if qty == int(qty) else qty
        except ValueError:
            qty = qty_lower  # keep as-is for 'several', 'few', etc.
    if isinstance(qty, float) and qty.is_integer():
        qty = int(qty)
    if unit_str.lower().startswith("week"):
        try:
            days = float(qty) * 7
            return f"{int(days) if days.is_integer() else days} days"
        except (TypeError, ValueError):
            pass
    if qty == 1 and isinstance(qty, int):
        unit = unit_str.lower().rstrip("s")
    else:
        unit = UNIT_NORMALIZATION.get(unit_str.lower().rstrip("s") + "s",
                                      unit_str.lower())
    # Handle plural unit input too
    unit = UNIT_NORMALIZATION.get(unit_str.lower(), unit)
    return f"{qty} {unit}".strip()
