"""
Evidence Linker Module: Maps extracted clinical facts and entities back to their exact
verbatim source sentences and character offsets for 100% auditability.
"""

from typing import List, Dict, Any
from src.utils import get_logger

logger = get_logger("EvidenceLinker")


class EvidenceLinker:
    """Links extracted structured entities back to source text citations."""

    def build_evidence_map(
        self,
        raw_entities: List[Dict[str, Any]],
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Constructs an explicit evidence record for every extracted entity.

        Returns:
            List of evidence dictionaries containing:
                - entity_text: Extracted entity
                - category: Medical entity category
                - context: Context status (CURRENT, PAST, NEGATED)
                - sentence_idx: 0-indexed sentence position
                - source_sentence: Exact verbatim sentence from document
                - start_char: Start offset in sentence
                - end_char: End offset in sentence
        """
        evidence_records = []

        for idx, ent in enumerate(raw_entities, 1):
            s_idx = ent.get("sentence_idx", 0)
            source_sent = sentences[s_idx] if s_idx < len(sentences) else ent.get("sentence_text", "")

            record = {
                "id": idx,
                "entity_text": ent["text"],
                "category": ent["category"],
                "context": ent["context"],
                "sentence_idx": s_idx + 1,  # 1-indexed for human display
                "source_sentence": source_sent,
                "start_char": ent.get("start_char", 0),
                "end_char": ent.get("end_char", len(ent["text"]))
            }
            evidence_records.append(record)

        return evidence_records

    def format_evidence_table_markdown(self, evidence_records: List[Dict[str, Any]]) -> str:
        """Formats evidence records as a Markdown table for display in Streamlit or reports."""
        if not evidence_records:
            return "No evidence traces available."

        lines = [
            "| # | Extracted Entity | Category | Context Tag | Source Sentence |",
            "|---|---|---|---|---|"
        ]

        for r in evidence_records:
            sent_highlight = r["source_sentence"].replace(
                r["entity_text"], f"**{r['entity_text']}**"
            )
            line = f"| {r['id']} | `{r['entity_text']}` | {r['category']} | `{r['context']}` | \"{sent_highlight}\" |"
            lines.append(line)

        return "\n".join(lines)
