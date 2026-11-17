"""
lexicons.py -- Clinical NER Lexicons and Regex Patterns
=======================================================

Defines all vocabulary sets and compiled regex patterns used by the rule-based
ClinicalNER engine.  Every term in a lexicon is stored **lower-cased** so the
engine only needs to lower-case the candidate surface form before lookup.

Regex patterns are kept as raw strings and compiled once at import time.
"""

import re
from typing import FrozenSet, List

# ---------------------------------------------------------------------------
# SYMPTOM
# ---------------------------------------------------------------------------

SYMPTOM_LEXICON: FrozenSet[str] = frozenset({
    # --- pain & discomfort ---
    "chest pain", "back pain", "abdominal pain", "joint pain", "ear pain",
    "eye pain", "pelvic pain", "neck pain", "leg pain", "arm pain",
    "muscle pain", "flank pain", "epigastric pain", "pain", "sob",
    # --- respiratory ---
    "shortness of breath", "breathlessness", "wheezing", "cough",
    "hemoptysis", "nasal congestion", "runny nose", "sore throat",
    # --- general / constitutional ---
    "fever", "chills", "sweats", "night sweats", "fatigue", "weakness",
    "weight loss", "loss of appetite", "anorexia", "malaise",
    # --- GI ---
    "nausea", "vomiting", "diarrhea", "constipation", "bloating",
    "heartburn", "dysphagia", "hematemesis", "melena",
    # --- neurological / musculoskeletal ---
    "headache", "dizziness", "vertigo", "syncope", "numbness", "tingling",
    "stiffness", "myalgia", "arthralgia",
    # --- cardiovascular ---
    "palpitations", "palpitation", "edema", "swelling",
    # --- skin / sensory ---
    "rash", "pruritus", "itching", "blurred vision", "diplopia",
    # --- urological / other ---
    "burning urination", "dysuria", "insomnia", "tremor",
})

# ---------------------------------------------------------------------------
# DISEASE
# ---------------------------------------------------------------------------

DISEASE_LEXICON: FrozenSet[str] = frozenset({
    # --- cardiovascular ---
    "hypertension", "htn", "coronary artery disease", "heart failure",
    "atrial fibrillation", "myocardial infarction", "stroke",
    "deep vein thrombosis", "peripheral artery disease",
    # --- metabolic / endocrine ---
    "diabetes", "dm", "t2dm", "diabetes mellitus", "type 2 diabetes", "type 1 diabetes",
    "hyperlipidemia", "hypothyroidism", "hyperthyroidism", "obesity",
    "metabolic syndrome",
    # --- respiratory ---
    "asthma", "copd", "pneumonia", "tuberculosis", "tb", "bronchitis",
    "pleural effusion", "pulmonary embolism",
    # --- infectious ---
    "malaria", "typhoid", "dengue", "sepsis", "meningitis", "hiv",
    "hepatitis b", "hepatitis c", "covid-19", "influenza",
    # --- GI ---
    "gerd", "peptic ulcer", "cirrhosis", "appendicitis",
    "inflammatory bowel disease", "crohn's disease", "ulcerative colitis",
    # --- renal ---
    "ckd", "chronic kidney disease", "acute kidney injury",
    "urinary tract infection", "nephrolithiasis",
    # --- musculoskeletal ---
    "osteoarthritis", "rheumatoid arthritis", "gout", "osteoporosis",
    "ankylosing spondylitis",
    # --- neurological / psychiatric ---
    "epilepsy", "migraine", "parkinson's disease", "alzheimer's disease",
    "depression", "anxiety", "schizophrenia",
    # --- haematological / oncological ---
    "anemia", "cancer", "leukemia", "lymphoma",
})

# ---------------------------------------------------------------------------
# MEDICATION
# ---------------------------------------------------------------------------

MEDICATION_LEXICON: FrozenSet[str] = frozenset({
    # --- antidiabetics ---
    "metformin", "insulin", "glibenclamide", "glipizide", "sitagliptin",
    "empagliflozin", "dapagliflozin",
    # --- antihypertensives ---
    "amlodipine", "lisinopril", "losartan", "metoprolol", "atenolol",
    "ramipril", "telmisartan", "valsartan", "enalapril", "nifedipine",
    "hydralazine", "spironolactone", "furosemide",
    # --- statins / lipid ---
    "atorvastatin", "rosuvastatin", "simvastatin",
    # --- analgesics / antipyretics ---
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "diclofenac",
    "naproxen", "celecoxib", "tramadol", "morphine", "codeine",
    # --- antibiotics ---
    "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline",
    "cotrimoxazole", "metronidazole", "fluconazole", "ceftriaxone",
    "clindamycin", "vancomycin", "gentamicin", "meropenem", "piperacillin",
    # --- respiratory ---
    "albuterol", "salbutamol", "theophylline", "montelukast", "tiotropium",
    # --- GI ---
    "omeprazole", "pantoprazole", "ranitidine", "ondansetron",
    # --- antiepileptics ---
    "phenytoin", "carbamazepine", "valproate", "levetiracetam", "gabapentin",
    # --- psychiatry ---
    "amitriptyline", "sertraline", "fluoxetine", "diazepam", "alprazolam",
    "haloperidol",
    # --- steroids / immunosuppressants ---
    "prednisone", "prednisolone", "hydrocortisone", "dexamethasone",
    "hydroxychloroquine",
    # --- anticoagulants ---
    "warfarin", "heparin", "clopidogrel", "rivaroxaban", "apixaban",
    # --- antiparasitic / antimalarial ---
    "chloroquine", "artemether", "lumefantrine",
    # --- thyroid ---
    "levothyroxine", "carbimazole",
    # --- antihistamines ---
    "cetirizine", "fexofenadine", "loratadine",
    # --- cardiac ---
    "digoxin", "bisoprolol", "diltiazem",
})

# ---------------------------------------------------------------------------
# BODY PART
# ---------------------------------------------------------------------------

BODY_PART_LEXICON: FrozenSet[str] = frozenset({
    # --- head / neck ---
    "head", "scalp", "face", "forehead", "temple", "eye", "eyes",
    "ear", "ears", "nose", "mouth", "throat", "neck",
    # --- thorax ---
    "chest", "lung", "lungs", "heart", "breast", "rib", "ribs",
    "diaphragm", "trachea", "bronchi",
    # --- abdomen ---
    "abdomen", "stomach", "liver", "gallbladder", "pancreas",
    "spleen", "kidney", "kidneys", "colon", "bowel", "intestine",
    "appendix", "bladder",
    # --- musculoskeletal ---
    "spine", "back", "shoulder", "arm", "elbow", "wrist", "hand",
    "fingers", "hip", "leg", "knee", "ankle", "foot", "feet",
    "joint", "joints", "muscle", "bone",
    # --- neurological / vascular ---
    "brain", "skin", "lymph node", "lymph nodes",
})

# ---------------------------------------------------------------------------
# TEST
# ---------------------------------------------------------------------------

TEST_LEXICON: FrozenSet[str] = frozenset({
    # --- imaging ---
    "chest x-ray", "cxr", "ct scan", "ct chest", "ct abdomen",
    "mri", "mri brain", "mri spine", "ultrasound", "usg",
    "pet scan", "bone scan", "mammogram", "x-ray",
    # --- cardiac ---
    "ecg", "ekg", "echocardiogram", "echo", "stress test",
    "holter monitor", "angiography",
    # --- blood panels ---
    "cbc", "complete blood count", "lipid panel", "hba1c",
    "blood glucose", "fasting blood glucose", "random blood glucose",
    "troponin", "creatinine", "urea", "bun", "electrolytes",
    "serum sodium", "serum potassium",
    # --- liver / renal / thyroid ---
    "liver function test", "lft", "renal function test", "rft",
    "thyroid function test", "tft", "tsh", "t3", "t4",
    # --- urine / microbiology ---
    "urinalysis", "urine culture", "blood culture", "sputum culture",
    "csf analysis", "biopsy",
    # --- infectious disease ---
    "dengue ns1", "widal test", "malaria rdt", "rapid antigen test",
    "covid pcr", "pcr",
    # --- other ---
    "blood pressure", "bp", "hb", "hemoglobin", "spo2", "peak flow", "spirometry",
})

# ---------------------------------------------------------------------------
# TEST RESULT
# ---------------------------------------------------------------------------

TEST_RESULT_LEXICON: FrozenSet[str] = frozenset({
    "elevated", "normal", "abnormal", "positive", "negative", "high", "low", "sugar",
    "reactive", "non-reactive", "nonreactive", "borderline", "trace",
    "3+", "2+", "1+", "4+", "mild", "moderate", "severe",
    "within normal limits", "wnl", "unremarkable", "pending", "not done",
})

# ---------------------------------------------------------------------------
# PROCEDURE
# ---------------------------------------------------------------------------

PROCEDURE_LEXICON: FrozenSet[str] = frozenset({
    "surgery", "operation", "biopsy", "intubation", "catheterization",
    "dialysis", "chemotherapy", "radiotherapy", "radiation therapy",
    "transfusion", "blood transfusion", "transplant", "appendectomy",
    "cholecystectomy", "coronary bypass", "angioplasty", "stenting",
    "endoscopy", "colonoscopy", "bronchoscopy", "lumbar puncture",
    "thoracentesis", "paracentesis", "tracheostomy", "gastroscopy",
    "laparoscopy", "laparotomy", "caesarean section", "c-section",
    "dental extraction", "incision and drainage",
})

# ---------------------------------------------------------------------------
# ALLERGEN
# ---------------------------------------------------------------------------

ALLERGEN_LEXICON: FrozenSet[str] = frozenset({
    "penicillin", "sulfa", "latex", "peanuts", "nsaids", "codeine",
    "shellfish", "eggs", "aspirin", "contrast dye", "morphine",
    "amoxicillin", "dust mites", "pollen", "cat dander", "bee sting",
    "milk", "wheat", "soy", "tree nuts", "ibuprofen", "sulfonamide",
    "cephalosporins", "erythromycin", "tetracycline", "chlorhexidine",
})

# ---------------------------------------------------------------------------
# SEVERITY
# ---------------------------------------------------------------------------

SEVERITY_LEXICON: FrozenSet[str] = frozenset({
    # qualitative
    "mild", "moderate", "severe", "extreme", "significant", "slight",
    "intense", "excruciating", "minimal", "maximal", "minor", "major",
    # character descriptors
    "sharp", "dull", "burning", "aching", "stabbing", "throbbing",
    "crushing", "tearing", "squeezing", "pressure-like",
    # grading systems
    "grade 1", "grade 2", "grade 3", "grade 4",
    # pain-scale (0/10 - 10/10)
    "0/10", "1/10", "2/10", "3/10", "4/10", "5/10",
    "6/10", "7/10", "8/10", "9/10", "10/10",
})

# ---------------------------------------------------------------------------
# FAMILY MEMBER
# ---------------------------------------------------------------------------

FAMILY_MEMBER_LEXICON: FrozenSet[str] = frozenset({
    "father", "mother", "brother", "sister", "son", "daughter",
    "grandfather", "grandmother", "uncle", "aunt", "cousin",
    "parent", "parents", "sibling", "siblings", "child", "children",
    "spouse", "husband", "wife", "maternal", "paternal",
    "grand father", "grand mother",
})

# ---------------------------------------------------------------------------
# DOSAGE  (regex)
# ---------------------------------------------------------------------------

DOSAGE_PATTERN: str = (
    r"\b\d+(?:\.\d+)?\s*"          # numeric amount (e.g. 500, 0.5)
    r"(?:mg|mcg|µg|g|ml|mL|L|IU|units?|tablet|tablets?|cap(?:sule)?s?|"
    r"drop|drops?|puff|puffs?|patch(?:es)?|suppository|suppositories)"
    r"\b"
)

# ---------------------------------------------------------------------------
# FREQUENCY (regex strings -- compiled in engine)
# ---------------------------------------------------------------------------

FREQUENCY_PATTERNS: List[str] = [
    # explicit English phrases
    r"\bonce\s+(?:a\s+day|daily)\b",
    r"\btwice\s+(?:a\s+day|daily)\b",
    r"\bthrice\s+(?:a\s+day|daily)\b",
    r"\bthree\s+times\s+(?:a\s+day|daily)\b",
    r"\bfour\s+times\s+(?:a\s+day|daily)\b",
    r"\bevery\s+\d+\s+hours?\b",
    r"\bevery\s+(?:other\s+)?day\b",
    r"\bweekly\b",
    r"\bbiweekly\b",
    r"\bmonthly\b",
    r"\bdaily\b",
    # Latin/medical abbreviations
    r"\b(?:o\.?d|od)\b",           # once daily
    r"\b(?:b\.?d|bd)\b",           # twice daily
    r"\b(?:t\.?d\.?s|tds)\b",      # three times daily
    r"\b(?:q\.?d\.?s|qds)\b",      # four times daily
    r"\b(?:b\.?i\.?d|bid)\b",      # twice a day
    r"\b(?:t\.?i\.?d|tid)\b",      # three times a day
    r"\b(?:q\.?i\.?d|qid)\b",      # four times a day
    r"\bprn\b",                    # as needed
    r"\bs\.?o\.?s\b",              # if needed (SOS)
    r"\bas\s+needed\b",
    r"\bat\s+bedtime\b",
    r"\bevery\s+(?:4|6|8|12)\s+hours?\b",
]

# ---------------------------------------------------------------------------
# DURATION (regex strings)
# ---------------------------------------------------------------------------

DURATION_PATTERNS: List[str] = [
    r"\bfor\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
    r"\b\d+[-\s](?:day|week|month|year)\s+(?:course|history|duration)\b",
    r"\b(?:since|over\s+the\s+(?:past|last))\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)\b",
    r"\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b",
    r"\b(?:lifelong|chronic|acute|subacute|recurrent|intermittent)\b",
]

# ---------------------------------------------------------------------------
# DATE (regex strings)
# ---------------------------------------------------------------------------

DATE_PATTERNS: List[str] = [
    # numeric dates
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
    # month name + day + optional year
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    # relative dates
    r"\b(?:today|yesterday|tomorrow)\b",
    r"\b(?:this|last|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year)\b",
    r"\b\d{4}\b",  # bare year (e.g. "in 2021")
]

# ---------------------------------------------------------------------------
# Pre-compiled regex objects (used directly by ner_engine)
# ---------------------------------------------------------------------------

COMPILED_DOSAGE: re.Pattern[str] = re.compile(DOSAGE_PATTERN, re.IGNORECASE)

COMPILED_FREQUENCY: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in FREQUENCY_PATTERNS
]

COMPILED_DURATION: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in DURATION_PATTERNS
]

COMPILED_DATE: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS
]

# Severity & test-result terms are looked up via lexicon but can also be
# captured via regex for the multi-word variants already in the frozensets.
COMPILED_SEVERITY: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(SEVERITY_LEXICON, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

COMPILED_TEST_RESULT: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(TEST_RESULT_LEXICON, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
