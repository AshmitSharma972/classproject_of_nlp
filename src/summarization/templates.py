"""
Summary templates for the clinical NLP pipeline.

All templates generate summaries exclusively from structured PatientRecord data.
No raw text is ever used directly in summary generation.

Template sections (in order):
  1. Present / Current Findings
  2. Negated Findings
  3. Historical Findings
  4. Current Medications
  5. Temporal / Timeline Information
  6. Uncertain / Possible Findings
  7. Disclaimer
"""

from __future__ import annotations

from typing import Dict, List

DISCLAIMER = (
    "\n[NOTICE] Decision support only. The physician remains responsible "
    "for clinical decisions.\n"
)

SECTION_HEADER = "-" * 60

# Section display ordering for the summary
SECTION_ORDER: List[str] = [
    "Chief Complaint",
    "HPI",
    "Past History",
    "Drug History",
    "Allergy",
    "Family History",
    "Personal History",
    "ROS",
]


def format_bullet_list(items: List[str], indent: int = 2) -> str:
    """Format a list of strings as a bulleted list."""
    pad = " " * indent
    # Defensive final rendering guard. Earlier stages retain every source
    # mention for evidence, but a summary must never render the same visible
    # fact more than once.
    unique_items: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(str(item).split()).casefold()
        if key and key not in seen:
            seen.add(key)
            unique_items.append(str(item))
    return "\n".join(f"{pad}* {item}" for item in unique_items)


def format_section_block(title: str, items: List[str]) -> str:
    """Format a titled section block."""
    if not items:
        return ""
    lines = [
        f"\n{SECTION_HEADER}",
        f"  {title.upper()}",
        SECTION_HEADER,
        format_bullet_list(items),
    ]
    return "\n".join(lines)


def build_summary_text(
    sections: Dict[str, List[str]],
    timeline_entries: List[str],
    record_id: str,
) -> str:
    """
    Build the complete summary text from structured data only.

    Args:
        sections: Dict of section_name → list of display strings.
        timeline_entries: List of timeline label strings.
        record_id: PatientRecord ID for traceability.

    Returns:
        The formatted clinical summary string.
    """
    lines: List[str] = [
        "=" * 60,
        "  CLINICAL HISTORY SUMMARY",
        f"  Record ID: {record_id}",
        "=" * 60,
    ]

    # 1. Chief Complaint
    cc = sections.get("Chief Complaint", [])
    if cc:
        lines.append(format_section_block("Chief Complaint", cc))

    # 2. History of Present Illness
    hpi = sections.get("HPI", [])
    if hpi:
        lines.append(format_section_block("History of Present Illness", hpi))

    # 3. Past Medical History
    past = sections.get("Past History", [])
    if past:
        lines.append(format_section_block("Past Medical History", past))

    # 4. Current Medications
    drugs = sections.get("Drug History", [])
    if drugs:
        lines.append(format_section_block("Current Medications", drugs))

    # 5. Allergies
    allergies = sections.get("Allergy", [])
    if allergies:
        lines.append(format_section_block("Allergies", allergies))

    # 6. Family History
    family = sections.get("Family History", [])
    if family:
        lines.append(format_section_block("Family History", family))

    # 7. Personal / Social History
    personal = sections.get("Personal History", [])
    if personal:
        lines.append(format_section_block("Personal / Social History", personal))

    # 8. Review of Systems (negated findings)
    ros = sections.get("ROS", [])
    if ros:
        lines.append(format_section_block("Review of Systems (Negated / Absent) — Pertinent Negatives", ros))

    # 9. Timeline
    if timeline_entries:
        lines.append(format_section_block("Clinical Timeline", timeline_entries))

    if not any(sections.get(name) for name in SECTION_ORDER) and not timeline_entries:
        lines.append(format_section_block("Summary", ["Insufficient data to generate a summary."]))

    # Disclaimer
    lines.append(f"\n{SECTION_HEADER}")
    lines.append(DISCLAIMER)
    lines.append(SECTION_HEADER)

    return "\n".join(lines)
