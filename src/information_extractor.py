"""
Information Extractor Module: Maps extracted medical entities and context tags into
structured clinical summary slots (Chief Complaints, Symptoms, History, Medications, etc.).
"""

from typing import List, Dict, Any
from src.medical_ner import MedicalNER
from src.utils import get_logger

logger = get_logger("InformationExtractor")


class InformationExtractor:
    """Categorizes raw NER extractions into structured patient clinical slots."""

    def __init__(self):
        self.ner_engine = MedicalNER()

    def process_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Runs Medical NER over sentences and groups entities into structured clinical categories.

        Returns:
            Dict containing raw_entities, structured_slots, and context_counts.
        """
        raw_entities = self.ner_engine.extract_entities(sentences)

        # Initialize structured category buckets
        structured_slots = {
            "Chief Complaints": [],
            "Symptoms": [],
            "Medical History": [],
            "Medications": [],
            "Allergies": [],
            "Tests and Results": [],
            "Important Findings": [],
            "Relevant Dates/Duration": [],
            "Negated Findings": []
        }

        # Track counts
        context_counts = {"CURRENT": 0, "PAST": 0, "NEGATED": 0}

        for ent in raw_entities:
            ctx = ent["context"]
            cat = ent["category"]
            txt = ent["text"]
            sentence = ent["sentence_text"]

            if ctx in context_counts:
                context_counts[ctx] += 1

            # Format item string with temporal/context badge
            item_str = f"{txt} [{ctx}]"

            # 1. Negated Entities
            if ctx == "NEGATED":
                structured_slots["Negated Findings"].append(f"No {txt} (denied/absent in source)")
                continue

            # 2. Past History Entities
            if ctx == "PAST" or cat == "Medical History":
                if txt not in structured_slots["Medical History"]:
                    structured_slots["Medical History"].append(f"{txt} (History)")
                continue

            # 3. Current Condition Entities
            if cat == "Symptoms":
                if "chief complaint" in sentence.lower() or "c/o" in sentence.lower():
                    if txt not in structured_slots["Chief Complaints"]:
                        structured_slots["Chief Complaints"].append(txt)
                else:
                    if txt not in structured_slots["Symptoms"]:
                        structured_slots["Symptoms"].append(txt)

            elif cat in ["Diseases/Conditions"]:
                if txt not in structured_slots["Medical History"]:
                    structured_slots["Medical History"].append(txt)

            elif cat in ["Medications", "Dosage"]:
                if txt not in structured_slots["Medications"]:
                    structured_slots["Medications"].append(txt)

            elif cat == "Allergies":
                if txt not in structured_slots["Allergies"]:
                    structured_slots["Allergies"].append(txt)

            elif cat in ["Medical Tests", "Test Results"]:
                test_finding = f"{txt} ({sentence.strip()})" if cat == "Test Results" else txt
                if test_finding not in structured_slots["Tests and Results"]:
                    structured_slots["Tests and Results"].append(test_finding)

            elif cat == "Relevant Dates/Duration":
                if txt not in structured_slots["Relevant Dates/Duration"]:
                    structured_slots["Relevant Dates/Duration"].append(txt)

            elif cat in ["Procedures", "Body Parts"]:
                finding = f"{cat[:-1] if cat.endswith('s') else cat}: {txt}"
                if finding not in structured_slots["Important Findings"]:
                    structured_slots["Important Findings"].append(finding)

        # Post-process: ensure Chief Complaints has fallback if empty but current symptoms exist
        if not structured_slots["Chief Complaints"] and structured_slots["Symptoms"]:
            structured_slots["Chief Complaints"].append(structured_slots["Symptoms"][0])

        return {
            "raw_entities": raw_entities,
            "structured_slots": structured_slots,
            "context_counts": context_counts
        }
