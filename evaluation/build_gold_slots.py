"""Build the initial synthetic clinician-review queue; never score TODO rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EN = (ROOT.parent / "tests" / "fixtures" / "english_narrations.txt").read_text(encoding="utf-8").splitlines()[:30]
HI = [
    "मुझे पाँच दिनों से बुखार है।", "मुझे खाँसी नहीं है।", "मुझे मधुमेह का इतिहास है।",
    "मेरी माँ को मधुमेह था।", "संभावित निमोनिया का उल्लेख है।", "सीने में दर्द से इनकार है।",
    "मुझे दो सप्ताह से सिरदर्द है।", "मुझे साँस लेने में तकलीफ़ है।", "मुझे कोई एलर्जी नहीं है।",
    "मेटफॉर्मिन 500 mg दिन में दो बार लेता हूँ।", "रक्तचाप 140/90 mmHg है।", "बुखार कल से है।",
    "मुझे उल्टी नहीं है लेकिन मतली है।", "पेनिसिलिन से एलर्जी है।", "मुझे उच्च रक्तचाप है।",
]
KN = [
    "ನನಗೆ ಐದು ದಿನಗಳಿಂದ ಜ್ವರ ಇದೆ.", "ನನಗೆ ಕೆಮ್ಮು ಇಲ್ಲ.", "ನನಗೆ ಮಧುಮೇಹದ ಇತಿಹಾಸವಿದೆ.",
    "ನನ್ನ ತಾಯಿಗೆ ಮಧುಮೇಹ ಇತ್ತು.", "ಸಂಭಾವ್ಯ ನ್ಯುಮೋನಿಯಾ ಎಂದು ಉಲ್ಲೇಖಿಸಲಾಗಿದೆ.", "ಎದೆನೋವನ್ನು ನಿರಾಕರಿಸುತ್ತಾರೆ.",
    "ಎರಡು ವಾರಗಳಿಂದ ತಲೆನೋವು ಇದೆ.", "ಉಸಿರಾಟದ ತೊಂದರೆ ಇದೆ.", "ಯಾವುದೇ ಅಲರ್ಜಿ ಇಲ್ಲ.",
    "ಮೆಟ್ಫಾರ್ಮಿನ್ 500 mg ದಿನಕ್ಕೆ ಎರಡು ಬಾರಿ ತೆಗೆದುಕೊಳ್ಳುತ್ತಾರೆ.", "ರಕ್ತದೊತ್ತಡ 140/90 mmHg ಇದೆ.", "ನಿನ್ನೆಿನಿಂದ ಜ್ವರ ಇದೆ.",
    "ವಾಂತಿ ಇಲ್ಲ ಆದರೆ ವಾಕರಿಕೆ ಇದೆ.", "ಪೆನಿಸಿಲಿನ್ ಅಲರ್ಜಿ ಇದೆ.", "ಅಧಿಕ ರಕ್ತದೊತ್ತಡ ಇದೆ.",
]

rows = []
for lang, texts in (("English", EN), ("Hindi", HI), ("Kannada", KN)):
    for index, text in enumerate(texts, 1):
        rows.append({"case_id": f"{lang[:2].lower()}-{index:02d}", "language": lang,
                     "text": text, "english_text": None, "annotations": None,
                     "review_status": "TODO: clinician annotation and approval"})
out = ROOT / "gold" / "clinical_cases.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    raise SystemExit(f"Refusing to overwrite existing review data: {out}")
out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
print(f"Wrote {len(rows)} clinician-review slots: English={len(EN)}, Hindi={len(HI)}, Kannada={len(KN)}")
