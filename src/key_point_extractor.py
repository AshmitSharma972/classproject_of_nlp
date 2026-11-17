"""
Key Point Extractor Module: Identifies and ranks clinically relevant sentences
using an extractive clinical priority scoring algorithm.
"""

from typing import List, Dict, Any
from config import CLINICAL_ABNORMAL_KEYWORDS
from src.utils import get_logger

logger = get_logger("KeyPointExtractor")


class KeyPointExtractor:
    """Ranks document sentences by clinical priority, abnormal findings, and severity."""

    def __init__(self):
        self.abnormal_keywords = set(CLINICAL_ABNORMAL_KEYWORDS)

    def extract_key_points(self, sentences: List[str], raw_entities: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates clinical priority scores for each sentence and returns top_n key points.

        Scoring heuristic:
        - Entity density: +2 points per medical entity in sentence
        - Current context: +3 points if entity context is CURRENT
        - Severity / Abnormal keywords: +4 points per abnormal keyword match
        - Allergy / Chief complaint sentence: +5 points
        - Sentence length penalty for extremely short or extremely long sentences
        """
        if not sentences:
            return []

        # Map entities to sentence indices
        sent_entity_map: Dict[int, List[Dict[str, Any]]] = {idx: [] for idx in range(len(sentences))}
        for ent in raw_entities:
            s_idx = ent.get("sentence_idx", 0)
            if s_idx in sent_entity_map:
                sent_entity_map[s_idx].append(ent)

        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = 0.0
            sent_lower = sentence.lower()
            entities = sent_entity_map.get(idx, [])

            # 1. Entity count & context score
            score += len(entities) * 2.0
            for ent in entities:
                if ent["context"] == "CURRENT":
                    score += 3.0
                elif ent["context"] == "PAST":
                    score += 1.0

            # 2. Abnormal / Critical findings bonus
            for kw in self.abnormal_keywords:
                if kw in sent_lower:
                    score += 4.0

            # 3. High priority section headers / keywords
            if any(term in sent_lower for term in ["chief complaint", "impression", "assessment", "diagnosis", "allergy", "allergic"]):
                score += 5.0

            # 4. Length penalty
            word_count = len(sentence.split())
            if word_count < 4 or word_count > 60:
                score *= 0.5

            scored_sentences.append({
                "sentence_idx": idx,
                "sentence": sentence,
                "score": round(score, 2),
                "entity_count": len(entities)
            })

        # Sort by score descending
        ranked = sorted(scored_sentences, key=lambda x: x["score"], reverse=True)
        return ranked[:top_n]
