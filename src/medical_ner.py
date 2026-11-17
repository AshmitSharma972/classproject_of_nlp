"""
Medical NER Module: Implements clinical named entity recognition combined with a
NegEx / ConText contextual algorithm for negation and temporal scope detection.
"""

import re
from typing import List, Dict, Any, Tuple
from config import NEGATION_TRIGGERS, PAST_HISTORY_TRIGGERS, CURRENT_CONDITION_TRIGGERS
from src.utils import get_logger

logger = get_logger("MedicalNER")

# Clinical Lexicons & Pattern Regex
SYMPTOM_LEXICON = {
    "fever", "feverish", "headache", "headaches", "cough", "coughing", "shortness of breath",
    "dyspnea", "chest pain", "chest tightness", "fatigue", "nausea", "vomiting", "diarrhea",
    "dizziness", "vertigo", "chills", "sweats", "night sweats", "myalgia", "arthralgia",
    "joint pain", "back pain", "lower back pain", "abdominal pain", "swelling", "edema",
    "palpitations", "wheezing", "rash", "pruritus", "numbness", "tingling", "weakness",
    "sore throat", "hemoptysis", "weight loss", "anorexia", "insomnia", "stiffness"
}

DISEASE_LEXICON = {
    "hypertension", "htn", "type 2 diabetes", "t2dm", "diabetes mellitus", "diabetes",
    "asthma", "pneumonia", "copd", "coronary artery disease", "cad", "heart failure",
    "chf", "myocardial infarction", "mi", "stroke", "cva", "osteoarthritis", "rheumatoid arthritis",
    "chronic kidney disease", "ckd", "hyperlipidemia", "atrial fibrillation", "afib",
    "gastroesophageal reflux", "gerd", "depression", "anxiety", "appendicitis", "bronchitis"
}

MEDICATION_LEXICON = {
    "metformin", "amlodipine", "lisinopril", "aspirin", "atorvastatin", "simvastatin",
    "amoxicillin", "azithromycin", "ibuprofen", "acetaminophen", "paracetamol", "losartan",
    "hydrochlorothiazide", "hctz", "metoprolol", "albuterol", "prednisone", "omeprazole",
    "pantoprazole", "gabapentin", "levothyroxine", "furosemide", "lasix", "insulin",
    "clopidogrel", "plavix", "warfarin", "heparin", "enoxaparin", "ciprofloxacin"
}

TEST_LEXICON = {
    "chest x-ray", "cxr", "ct scan", "mri", "ultrasound", "ecg", "ekg", "echocardiogram",
    "hba1c", "a1c", "cbc", "complete blood count", "bmp", "cmp", "lipid panel",
    "troponin", "bnp", "d-dimer", "creatinine", "blood glucose", "urinalysis", "biopsy",
    "blood pressure", "bp", "heart rate", "pulse", "oxygen saturation", "spo2", "temperature"
}

PROCEDURE_LEXICON = {
    "appendectomy", "cholecystectomy", "pci", "angioplasty", "cabg", "coronary artery bypass",
    "intubation", "colonoscopy", "endoscopy", "hemodialysis", "dialysis", "biopsy",
    "lumbar puncture", "catheterization", "pacemaker implantation", "knee replacement",
    "hip replacement", "suturing", "debridement", "tracheostomy"
}

ALLERGY_LEXICON = {
    "penicillin", "sulfa", "sulfonamides", "latex", "peanuts", "nsaids", "codeine",
    "morphine", "contrast dye", "aspirin allergy", "shellfish", "eggs", "tretinoin"
}

BODY_PART_LEXICON = {
    "chest", "lungs", "heart", "abdomen", "stomach", "liver", "kidney", "kidneys",
    "brain", "head", "neck", "throat", "back", "spine", "left arm", "right arm",
    "left leg", "right leg", "knee", "left knee", "right knee", "shoulder", "skin",
    "eyes", "ears", "colon", "gallbladder", "pancreas", "spleen", "bladder"
}


class MedicalNER:
    """Clinical Named Entity Recognition & Contextual Negation/Temporal Tagging Engine."""

    def __init__(self):
        self._compile_regex_patterns()

    def _compile_regex_patterns(self):
        """Compiles regular expressions for dosages, dates/durations, and clinical metrics."""
        # Dosage pattern e.g. 500 mg, 10mg daily, 500mg BID, 100 mcg
        self.dosage_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units|u|tablets?|capsules?|pills?)"
            r"(?:\s*(?:once|twice|thrice|daily|bid|tid|qid|prn|po|iv|subq|every\s*\d+\s*hours?))?\b",
            re.IGNORECASE
        )

        # Dates / Duration pattern e.g. 3 days, last week, 2024-05-12, for 2 months, 6 years ago
        self.date_duration_pattern = re.compile(
            r"\b(?:\d+\s*(?:days?|weeks?|months?|years?|hours?)(?:\s*ago|\s*duration)?|"
            r"last\s*(?:week|month|year|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
            r"\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            re.IGNORECASE
        )

        # Test results pattern e.g. 8.5%, elevated, 140/90 mmHg, normal, positive, negative
        self.test_result_pattern = re.compile(
            r"\b(?:\d+(?:\.\d+)?\s*(?:%|mg/dl|g/dl|mmol/l|mEq/L|bpm|mmHg)|"
            r"\d{2,3}/\d{2,3}\s*mmHg|"
            r"elevated|normal|abnormal|positive|negative|high|low|markedly increased|unremarkable)\b",
            re.IGNORECASE
        )

    def extract_entities(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extracts medical entities from a list of sentences and tags each entity with its
        contextual status (CURRENT, PAST, NEGATED) using NegEx/ConText logic.
        """
        extracted_entities = []

        for sent_idx, sentence in enumerate(sentences):
            sent_entities = self._extract_entities_from_sentence(sentence, sent_idx)
            extracted_entities.extend(sent_entities)

        return extracted_entities

    def _extract_entities_from_sentence(self, sentence: str, sent_idx: int) -> List[Dict[str, Any]]:
        """Scans a single sentence for lexicon matches and regex entities, resolving context."""
        entities = []
        sent_lower = sentence.lower()

        # Helper match function
        def add_entity(entity_text: str, category: str, start_char: int, end_char: int):
            context = self._determine_context(sentence, start_char, end_char)
            entities.append({
                "text": entity_text,
                "category": category,
                "context": context,  # CURRENT, PAST, or NEGATED
                "sentence_idx": sent_idx,
                "sentence_text": sentence,
                "start_char": start_char,
                "end_char": end_char
            })

        # 1. Lexicon Scans (Symptoms, Diseases, Medications, Tests, Procedures, Allergies, Body Parts)
        for term in SYMPTOM_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Symptoms", m.start(), m.end())

        for term in DISEASE_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Diseases/Conditions", m.start(), m.end())

        for term in MEDICATION_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Medications", m.start(), m.end())

        for term in TEST_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Medical Tests", m.start(), m.end())

        for term in PROCEDURE_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Procedures", m.start(), m.end())

        for term in ALLERGY_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Allergies", m.start(), m.end())

        for term in BODY_PART_LEXICON:
            for m in re.finditer(r"\b" + re.escape(term) + r"\b", sent_lower):
                add_entity(sentence[m.start():m.end()], "Body Parts", m.start(), m.end())

        # 2. Regex Scans (Dosage, Dates/Duration, Test Results)
        for m in self.dosage_pattern.finditer(sentence):
            add_entity(m.group(), "Dosage", m.start(), m.end())

        for m in self.date_duration_pattern.finditer(sentence):
            add_entity(m.group(), "Relevant Dates/Duration", m.start(), m.end())

        for m in self.test_result_pattern.finditer(sentence):
            # Avoid duplicate tagging if captured under symptoms
            if not any(e["start_char"] == m.start() and e["category"] == "Test Results" for e in entities):
                add_entity(m.group(), "Test Results", m.start(), m.end())

        # 3. Deduplicate overlapping identical extractions
        deduped = self._deduplicate_entities(entities)
        return deduped

    def _determine_context(self, sentence: str, start_char: int, end_char: int) -> str:
        """
        Applies NegEx / ConText algorithm to determine whether entity at [start_char, end_char]
        is NEGATED, PAST, or CURRENT.
        """
        sent_lower = sentence.lower()

        # Scope window: look up to 6 words before the entity and 4 words after
        prefix_window = sent_lower[max(0, start_char - 45):start_char]
        suffix_window = sent_lower[end_char:min(len(sent_lower), end_char + 30)]

        # Check Negation Triggers
        for trigger in NEGATION_TRIGGERS:
            if re.search(r"\b" + re.escape(trigger) + r"\b", prefix_window):
                return "NEGATED"
            if re.search(r"\b" + re.escape(trigger) + r"\b", suffix_window):
                return "NEGATED"

        # Check Past History Triggers
        for trigger in PAST_HISTORY_TRIGGERS:
            if re.search(r"\b" + re.escape(trigger) + r"\b", prefix_window):
                return "PAST"

        # Check Current Triggers
        for trigger in CURRENT_CONDITION_TRIGGERS:
            if re.search(r"\b" + re.escape(trigger) + r"\b", prefix_window):
                return "CURRENT"

        # Default to CURRENT if no negation/past trigger is present
        return "CURRENT"

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters out duplicate entity matches within the same sentence."""
        seen = set()
        unique_entities = []
        for e in sorted(entities, key=lambda x: (x["start_char"], -len(x["text"]))):
            key = (e["text"].lower(), e["category"], e["context"], e["start_char"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        return unique_entities
