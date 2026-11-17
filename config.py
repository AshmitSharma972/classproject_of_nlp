"""
Configuration Module for AI-Based Medical Document Summarization System.
Centralizes paths, model names, entity taxonomies, negation/temporal triggers,
and clinical scoring parameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_documents"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_SUMMARIES_FILE = DATA_DIR / "reference_summaries.json"

# PDF Extraction Config
MIN_CHAR_COUNT_PER_PAGE = 40  # Below this, PDF is flagged as scanned requiring OCR

# Summarization Models
PRIMARY_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
FALLBACK_SUMMARIZATION_MODEL = "t5-small"
DEFAULT_MAX_SUMMARY_LENGTH = 200
DEFAULT_MIN_SUMMARY_LENGTH = 50

# Medical Entity Categories
ENTITY_CATEGORIES = [
    "Chief Complaints",
    "Symptoms",
    "Diseases/Conditions",
    "Medications",
    "Dosage",
    "Medical Tests",
    "Test Results",
    "Procedures",
    "Allergies",
    "Medical History",
    "Relevant Dates/Duration",
    "Body Parts",
]

# Negation and Context Triggers (ConText / NegEx inspired)
NEGATION_TRIGGERS = [
    "no", "not", "denies", "without", "absent", "negative for", "rules out",
    "ruled out", "free of", "unremarkable for", "no evidence of", "no sign of",
    "negated", "denied", "neither", "nor", "clear of"
]

PAST_HISTORY_TRIGGERS = [
    "history of", "past history", "prior to", "previously", "years ago",
    "months ago", "former", "past medical history", "pmh", "recollected",
    "known case of", "treated for in the past", "status post", "s/p"
]

CURRENT_CONDITION_TRIGGERS = [
    "currently", "presenting with", "today", "now", "chief complaint",
    "admits to", "complains of", "c/o", "presents with", "active", "acute",
    "on presentation", "current"
]

# Clinical Priority Keywords (for Key Point Detection)
CLINICAL_ABNORMAL_KEYWORDS = [
    "elevated", "high", "low", "abnormal", "critical", "positive", "severe",
    "markedly", "acute", "worsening", "persistent", "refractory", "decreased",
    "increased", "pathological", "lesion", "mass", "hemorrhage", "ischemic",
    "tachycardia", "bradycardia", "hypotension", "hypertension", "febrile"
]

# Medical Disclaimer Text
MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This system is an AI-based clinical information assistance tool "
    "developed strictly for educational and research purposes. It DOES NOT provide "
    "medical diagnoses, clinical decisions, or treatment recommendations. "
    "All summaries and extractions must be reviewed and verified by a qualified healthcare professional."
)
