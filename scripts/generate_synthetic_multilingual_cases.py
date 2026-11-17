"""Create a de-identified multilingual synthetic clinical-note corpus.

The output is deliberately synthetic and must not be represented as hospital
records or used for clinical decision making.  Each record contains a 22-line
note and two aligned targets: a same-language summary and an English summary.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SEED = 20261025
COUNTS = {"en": 4000, "hi": 3000, "kn": 3000}

CONDITIONS = [
    ("acute febrile illness", "fever", "paracetamol", "complete blood count"),
    ("type 2 diabetes mellitus", "increased thirst", "metformin", "HbA1c"),
    ("essential hypertension", "headache", "amlodipine", "blood pressure monitoring"),
    ("acute bronchitis", "cough", "azithromycin", "chest radiograph"),
    ("urinary tract infection", "burning urination", "nitrofurantoin", "urine culture"),
    ("gastritis", "upper abdominal pain", "pantoprazole", "stool occult blood test"),
    ("migraine", "headache", "sumatriptan", "neurological examination"),
    ("iron deficiency anemia", "fatigue", "ferrous sulfate", "hemoglobin"),
    ("osteoarthritis", "knee pain", "diclofenac gel", "knee radiograph"),
    ("allergic rhinitis", "sneezing", "cetirizine", "allergy assessment"),
]

NAMES = {
    "en": ["Aarav", "Meera", "Rohan", "Kavya", "Vikram", "Ananya"],
    "hi": ["आरव", "मीरा", "रोहन", "काव्या", "विक्रम", "अनन्या"],
    "kn": ["ಆರವ್", "ಮೀರಾ", "ರೋಹನ್", "ಕಾವ್ಯಾ", "ವಿಕ್ರಮ್", "ಅನನ್ಯಾ"],
}


def english_note(case_id: str, rng: random.Random, age: int, sex: str, condition: tuple[str, str, str, str]) -> tuple[str, str]:
    diagnosis, complaint, medicine, test = condition
    name = rng.choice(NAMES["en"])
    duration = rng.randint(2, 14)
    dose = rng.choice(["once daily", "twice daily", "after meals"])
    lines = [
        f"SYNTHETIC TRAINING CASE: {case_id}",
        "This is a fictional, de-identified educational record.",
        f"Patient name: {name} (synthetic identifier).",
        f"Age: {age} years.",
        f"Sex: {sex}.",
        "Encounter type: outpatient clinical assessment.",
        f"Chief complaint: {complaint} for {duration} days.",
        f"History of present illness: symptoms are consistent with {diagnosis}.",
        "Symptoms have been reviewed with the patient.",
        "No immediate emergency symptoms were reported in this synthetic scenario.",
        "Past medical history was reviewed.",
        "No known drug allergies were reported.",
        "Family history is non-contributory in this synthetic record.",
        "Social history: no tobacco use reported.",
        "General examination completed.",
        "Patient is clinically stable for outpatient management.",
        f"Investigation ordered: {test}.",
        f"Assessment: probable {diagnosis}.",
        f"Medication plan: {medicine}, {dose}, as clinically appropriate.",
        "Supportive care and hydration were discussed.",
        "Return precautions were explained.",
        "Follow-up is advised in 3 to 7 days or earlier if symptoms worsen.",
    ]
    summary = (
        f"Synthetic case: {age}-year-old {sex.lower()} with {complaint} for {duration} days. "
        f"Assessment is probable {diagnosis}. {test} was ordered. "
        f"Plan includes {medicine} {dose}, supportive care, and follow-up."
    )
    return "\n".join(lines), summary


def hindi_note(case_id: str, rng: random.Random, age: int, sex: str, condition: tuple[str, str, str, str]) -> tuple[str, str, str]:
    diagnosis, complaint, medicine, test = condition
    name = rng.choice(NAMES["hi"])
    days = rng.randint(2, 14)
    sex_hi = "पुरुष" if sex == "Male" else "महिला"
    lines = [
        f"कृत्रिम प्रशिक्षण मामला: {case_id}", "यह काल्पनिक और पहचान-रहित शैक्षिक रिकॉर्ड है।",
        f"रोगी का नाम: {name} (कृत्रिम पहचान)।", f"आयु: {age} वर्ष।", f"लिंग: {sex_hi}।",
        "मुलाकात का प्रकार: बाह्य रोगी नैदानिक मूल्यांकन।", f"मुख्य शिकायत: {complaint} {days} दिनों से है।",
        f"वर्तमान बीमारी का इतिहास: लक्षण {diagnosis} के अनुरूप हैं।", "रोगी के साथ लक्षणों की समीक्षा की गई।",
        "इस कृत्रिम परिदृश्य में तत्काल आपात लक्षण नहीं बताए गए।", "पूर्व चिकित्सा इतिहास की समीक्षा की गई।",
        "दवा से किसी ज्ञात एलर्जी की सूचना नहीं है।", "इस कृत्रिम रिकॉर्ड में पारिवारिक इतिहास महत्वपूर्ण नहीं है।",
        "सामाजिक इतिहास: तंबाकू सेवन की सूचना नहीं है।", "सामान्य परीक्षण पूरा किया गया।",
        "रोगी बाह्य रोगी प्रबंधन के लिए चिकित्सकीय रूप से स्थिर है।", f"जांच का आदेश: {test}।",
        f"आकलन: संभावित {diagnosis}।", f"दवा योजना: {medicine}, चिकित्सकीय सलाह के अनुसार।",
        "सहायक देखभाल और पर्याप्त तरल पदार्थ लेने पर चर्चा की गई।", "वापस आने की सावधानियां समझाई गईं।",
        "लक्षण बढ़ने पर शीघ्र या 3 से 7 दिनों में फॉलो-अप की सलाह दी गई।",
    ]
    native = f"कृत्रिम मामला: {age} वर्षीय {sex_hi} को {days} दिनों से {complaint} है। संभावित निदान {diagnosis} है। {test} का आदेश दिया गया। फॉलो-अप की सलाह दी गई।"
    _, english = english_note(case_id, rng, age, sex, condition)
    return "\n".join(lines), native, english


def kannada_note(case_id: str, rng: random.Random, age: int, sex: str, condition: tuple[str, str, str, str]) -> tuple[str, str, str]:
    diagnosis, complaint, medicine, test = condition
    name = rng.choice(NAMES["kn"])
    days = rng.randint(2, 14)
    sex_kn = "ಪುರುಷ" if sex == "Male" else "ಮಹಿಳೆ"
    lines = [
        f"ಕೃತಕ ತರಬೇತಿ ಪ್ರಕರಣ: {case_id}", "ಇದು ಕಾಲ್ಪನಿಕ ಮತ್ತು ಗುರುತು-ತೆಗೆದ ಶೈಕ್ಷಣಿಕ ದಾಖಲೆ.",
        f"ರೋಗಿಯ ಹೆಸರು: {name} (ಕೃತಕ ಗುರುತು).", f"ವಯಸ್ಸು: {age} ವರ್ಷ.", f"ಲಿಂಗ: {sex_kn}.",
        "ಭೇಟಿಯ ವಿಧ: ಹೊರರೋಗಿ ವೈದ್ಯಕೀಯ ಮೌಲ್ಯಮಾಪನ.", f"ಮುಖ್ಯ ದೂರು: {complaint} {days} ದಿನಗಳಿಂದ ಇದೆ.",
        f"ಪ್ರಸ್ತುತ ಅನಾರೋಗ್ಯದ ಇತಿಹಾಸ: ಲಕ್ಷಣಗಳು {diagnosis} ಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತವೆ.", "ರೋಗಿಯೊಂದಿಗೆ ಲಕ್ಷಣಗಳನ್ನು ಪರಿಶೀಲಿಸಲಾಗಿದೆ.",
        "ಈ ಕೃತಕ ಸನ್ನಿವೇಶದಲ್ಲಿ ತಕ್ಷಣದ ತುರ್ತು ಲಕ್ಷಣಗಳು ವರದಿಯಾಗಿಲ್ಲ.", "ಹಿಂದಿನ ವೈದ್ಯಕೀಯ ಇತಿಹಾಸವನ್ನು ಪರಿಶೀಲಿಸಲಾಗಿದೆ.",
        "ತಿಳಿದಿರುವ ಔಷಧ ಅಲರ್ಜಿಗಳು ವರದಿಯಾಗಿಲ್ಲ.", "ಈ ಕೃತಕ ದಾಖಲೆಯಲ್ಲಿ ಕುಟುಂಬದ ಇತಿಹಾಸ ಮಹತ್ವದ್ದಲ್ಲ.",
        "ಸಾಮಾಜಿಕ ಇತಿಹಾಸ: ತಂಬಾಕು ಬಳಕೆ ವರದಿಯಾಗಿಲ್ಲ.", "ಸಾಮಾನ್ಯ ಪರೀಕ್ಷೆ ಪೂರ್ಣಗೊಂಡಿದೆ.",
        "ಹೊರರೋಗಿ ನಿರ್ವಹಣೆಗೆ ರೋಗಿ ವೈದ್ಯಕೀಯವಾಗಿ ಸ್ಥಿರರಾಗಿದ್ದಾರೆ.", f"ಆದೇಶಿಸಿದ ಪರೀಕ್ಷೆ: {test}.",
        f"ಮೌಲ್ಯಮಾಪನ: ಸಂಭವನೀಯ {diagnosis}.", f"ಔಷಧ ಯೋಜನೆ: {medicine}, ವೈದ್ಯಕೀಯ ಸಲಹೆಯಂತೆ.",
        "ಬೆಂಬಲಿತ ಆರೈಕೆ ಮತ್ತು ಸಾಕಷ್ಟು ದ್ರವ ಸೇವನೆ ಕುರಿತು ಚರ್ಚಿಸಲಾಗಿದೆ.", "ಮರಳಿ ಬರುವ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ವಿವರಿಸಲಾಗಿದೆ.",
        "ಲಕ್ಷಣಗಳು ಹೆಚ್ಚಾದರೆ ಬೇಗ ಅಥವಾ 3 ರಿಂದ 7 ದಿನಗಳಲ್ಲಿ ಅನುಸರಣೆ ಮಾಡಲು ಸಲಹೆ ನೀಡಲಾಗಿದೆ.",
    ]
    native = f"ಕೃತಕ ಪ್ರಕರಣ: {age} ವರ್ಷದ {sex_kn} ರೋಗಿಗೆ {days} ದಿನಗಳಿಂದ {complaint} ಇದೆ. ಸಂಭವನೀಯ ನಿರ್ಣಯ {diagnosis}. {test} ಆದೇಶಿಸಲಾಗಿದೆ. ಅನುಸರಣೆ ಮಾಡಲು ಸಲಹೆ ನೀಡಲಾಗಿದೆ."
    _, english = english_note(case_id, rng, age, sex, condition)
    return "\n".join(lines), native, english


def build_record(language: str, index: int, rng: random.Random) -> dict:
    case_id = f"SYN-{language.upper()}-{index:05d}"
    age, sex, condition = rng.randint(18, 85), rng.choice(["Male", "Female"]), rng.choice(CONDITIONS)
    if language == "en":
        source, summary = english_note(case_id, rng, age, sex, condition)
        summary_en = summary
    elif language == "hi":
        source, summary, summary_en = hindi_note(case_id, rng, age, sex, condition)
    else:
        source, summary, summary_en = kannada_note(case_id, rng, age, sex, condition)
    return {"case_id": case_id, "language": language, "source_text": source,
            "summary": summary, "summary_en": summary_en, "synthetic": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_multilingual_clinical_cases.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    with args.output.open("w", encoding="utf-8") as file:
        for language, count in COUNTS.items():
            for index in range(1, count + 1):
                file.write(json.dumps(build_record(language, index, rng), ensure_ascii=False) + "\n")
    print(f"Wrote {sum(COUNTS.values()):,} synthetic cases to {args.output}")


if __name__ == "__main__":
    main()
