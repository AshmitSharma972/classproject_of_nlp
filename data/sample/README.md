# Sample Dataset — Clinical NLP Pipeline

> **⚠️ SYNTHETIC DATA NOTICE**
> All 20 patient cases in this directory are **entirely synthetic**.
> They contain **no real Protected Health Information (PHI)**, no real patient identifiers,
> and no real medical records. This dataset was generated solely for software
> development, algorithm testing, and educational purposes.

---

## Purpose

This dataset provides a controlled, labelled set of 20 plain-English patient
narrations that exercise every major NLP annotation task in the clinical pipeline:

| Task | Description |
|---|---|
| **Named Entity Recognition (NER)** | Symptoms, diseases, medications, procedures, lab values, findings |
| **Assertion Classification** | PRESENT · NEGATED · POSSIBLE · HISTORICAL · FAMILY_HISTORY · ALLERGY |
| **Temporal Extraction** | Durations, relative dates, absolute dates, time-of-day, frequency |
| **Relation Extraction** | SYMPTOM-DURATION · MEDICATION-INDICATION · DISEASE-DATE · etc. |
| **Section Detection** | Chief Complaint · ROS · Medical History · Medications · Labs · etc. |
| **Coreference / Anaphora** | Resolution of "the fever", "similar problem last year" (case 15) |

---

## Case Coverage Table

| Case | File | Primary Clinical Scenario | Key NLP Challenges |
|------|------|--------------------------|-------------------|
| 01 | `case_01.txt` | Simple fever + cough | Basic negation, duration temporal |
| 02 | `case_02.txt` | Diabetes + hypertension + medications | Drug-dose-frequency triples, lab values |
| 03 | `case_03.txt` | Family history dominant | FAMILY_HISTORY assertion scope |
| 04 | `case_04.txt` | Uncertain / possible diagnosis | POSSIBLE modality ("possibly", "rule out") |
| 05 | `case_05.txt` | Dense negations | Multi-trigger negation ("denies", "no", "never") |
| 06 | `case_06.txt` | Medication dosage + frequency + duration | Med-dose-freq-duration 4-tuples |
| 07 | `case_07.txt` | Lab tests + numeric results | LAB_TEST → LAB_VALUE linking, units |
| 08 | `case_08.txt` | Surgical procedures + dates | PROCEDURE-DATE relations, past tense |
| 09 | `case_09.txt` | Historical disease (recovered TB) | HISTORICAL vs. PRESENT distinction |
| 10 | `case_10.txt` | Drug allergies + reactions | ALLERGY assertion, drug-reaction pairs |
| 11 | `case_11.txt` | Multi-symptom with severity scores | Severity NRS (8/10), modifier (mild/severe) |
| 12 | `case_12.txt` | ROS-style systematic review | High-density negation across organ systems |
| 13 | `case_13.txt` | Complex mixed (all categories) | Multi-label: history + family + meds + negation |
| 14 | `case_14.txt` | Temporal complexity (multi-step timeline) | Chained relative timestamps |
| 15 | `case_15.txt` | Anaphora / reference resolution | "the fever worsened", "similar problem last year" |
| 16 | `case_16.txt` | COPD + inhaler therapy | PRN frequency, COPD exacerbation, O₂ saturation |
| 17 | `case_17.txt` | Pediatric (parent-narrated) | Age attribute, narrator detection |
| 18 | `case_18.txt` | Psychiatric history + safety | Psychiatric assertion (suicidal ideation negated) |
| 19 | `case_19.txt` | Cardiac: CAD + CABG + polypharmacy | Procedure + 4-drug regimen, echo result |
| 20 | `case_20.txt` | Complex negation + possible + family history | Negated family history, unconfirmed family dx |

---

## Ground Truth Format (`ground_truth.json`)

```json
{
  "case_NN": {
    "entities": [
      {"text": "<surface text>", "type": "<ENTITY_TYPE>"}
    ],
    "assertions": [
      {"entity_text": "<surface text>", "status": "<ASSERTION_STATUS>"}
    ],
    "temporal": [
      {"raw_value": "<raw span>", "type": "<TEMPORAL_TYPE>", "normalized_value": "<ISO/custom>"}
    ],
    "relations": [
      {"head": "<entity>", "tail": "<entity or value>", "type": "<RELATION_TYPE>"}
    ],
    "medications": [
      {"drug": "<name>", "dose": "<dose>", "frequency": "<freq>", "duration": "<dur>", "status": "<CURRENT|PRIOR>"}
    ],
    "lab_results": [
      {"test": "<test name>", "value": "<numeric+unit>", "interpretation": "<low|normal|high>"}
    ],
    "family_history": [
      {"relation": "<family member>", "condition": "<disease>", "status": "<PRESENT|NEGATED|POSSIBLE>"}
    ],
    "allergies": [
      {"drug": "<drug>", "reaction": "<description>", "severity": "<severity>"}
    ],
    "sections": {
      "<Section Name>": ["<entity1>", "<entity2>"]
    }
  }
}
```

### Entity Types

| Type | Examples |
|---|---|
| `SYMPTOM` | fever, cough, chest pain, vomiting |
| `DISEASE` | diabetes, COPD, coronary artery disease |
| `MEDICATION` | metformin, salbutamol, aspirin |
| `PROCEDURE` | appendectomy, CABG, colonoscopy |
| `LAB_TEST` | CBC, HbA1c, fasting blood sugar |
| `LAB_VALUE` | hemoglobin, MCV, serum ferritin |
| `FINDING` | ST-segment elevation, tonsillar enlargement |
| `SOCIAL_HISTORY` | smoking |

### Assertion Statuses

| Status | Meaning |
|---|---|
| `PRESENT` | Affirmed, currently true |
| `NEGATED` | Explicitly denied |
| `POSSIBLE` | Uncertain / hypothetical |
| `HISTORICAL` | True in the past, not necessarily now |
| `FAMILY_HISTORY` | Observed in a family member |
| `NEGATED_FAMILY_HISTORY` | Explicitly absent in family |
| `POSSIBLE_FAMILY_HISTORY` | Uncertain family diagnosis |
| `ALLERGY` | Drug causing an allergic reaction |
| `ALLERGY_REACTION` | The reaction caused by an allergen |
| `INTOLERANCE` | Drug intolerance (non-allergic) |

### Temporal Types

| Type | Format | Example |
|---|---|---|
| `DURATION` | ISO 8601 Period | `P5D`, `P2W`, `P18M` |
| `RELATIVE_TIME` | `±<N><unit>` | `-3D`, `-6M`, `-1Y` |
| `ABSOLUTE_DATE` | `YYYY` or `YYYY-MM` | `2019`, `2018-03` |
| `AGE_AT_ONSET` | `<N>Y` | `24Y`, `62Y` |
| `FREQUENCY` | Clinical abbrev | `Q6H`, `BD`, `TID` |
| `TIME_OF_DAY` | descriptive | `night`, `morning` |

---

## Notes on Public Datasets

The following public datasets are commonly used for clinical NLP benchmarking.
**They are NOT included here** due to licensing restrictions, but are recommended
for model evaluation once proper DUA/access is obtained.

| Dataset | Task | License / Access |
|---|---|---|
| **n2c2 / i2b2** | NER, assertion, coreference, temporal | Requires DUA from Harvard DBMI; available at [n2c2.dbmi.hms.harvard.edu](https://n2c2.dbmi.hms.harvard.edu) |
| **NCBI Disease Corpus** | Disease NER | Freely available (PubMed abstracts); [NCBI FTP](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) |
| **BC5CDR** | Chemical + Disease NER; Chem-Disease relations | Available via BioCreative; [BC5CDR page](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/) |
| **MTSamples** | Clinical narrative diversity | Freely browsable at [mtsamples.com](https://www.mtsamples.com/); scraping may violate ToS — verify before use |

> **Important**: Using n2c2/i2b2 data requires a signed Data Use Agreement.
> Do not distribute, share, or commit these datasets to version control.
> Place any licensed corpora under `data/licensed/` (already listed in `.gitignore`).

---

## How to Run Evaluation Against Ground Truth

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline on sample cases

```bash
python -m src.pipeline.run \
    --input  data/sample/ \
    --output data/output/ \
    --pattern "case_*.txt"
```

### 3. Evaluate against ground truth

```bash
python -m src.evaluation.evaluate \
    --ground-truth data/sample/ground_truth.json \
    --predictions  data/output/predictions.json \
    --tasks entities assertions temporal relations \
    --report-dir  reports/
```

### 4. Expected output

```
============================================================
EVALUATION REPORT — Clinical NLP Pipeline
============================================================
Task            Precision   Recall   F1-Score
------------------------------------------------------------
entities        0.XXX       0.XXX    0.XXX
assertions      0.XXX       0.XXX    0.XXX
temporal        0.XXX       0.XXX    0.XXX
relations       0.XXX       0.XXX    0.XXX
------------------------------------------------------------
Full report saved to: reports/eval_YYYYMMDD_HHMMSS.json
```

### 5. Per-case debugging

```bash
python -m src.evaluation.evaluate \
    --ground-truth data/sample/ground_truth.json \
    --predictions  data/output/predictions.json \
    --case case_04 \
    --verbose
```

---

## File Listing

```
data/sample/
├── case_01.txt   ← Fever + cough, basic negation, temporal
├── case_02.txt   ← Diabetes + hypertension + medications
├── case_03.txt   ← Family history dominant
├── case_04.txt   ← Possible / uncertain diagnosis
├── case_05.txt   ← Dense negations
├── case_06.txt   ← Medication dose + frequency + duration
├── case_07.txt   ← Lab tests + numeric results
├── case_08.txt   ← Procedures + dates
├── case_09.txt   ← Historical disease (recovered TB)
├── case_10.txt   ← Drug allergies + reactions
├── case_11.txt   ← Multi-symptom with severity scores
├── case_12.txt   ← ROS-style systematic review
├── case_13.txt   ← Complex mixed (all categories)
├── case_14.txt   ← Temporal complexity (multi-step)
├── case_15.txt   ← Anaphora / reference resolution
├── case_16.txt   ← COPD + inhaler therapy
├── case_17.txt   ← Pediatric, parent-narrated
├── case_18.txt   ← Psychiatric + safety assessment
├── case_19.txt   ← Cardiac: CAD + CABG + polypharmacy
├── case_20.txt   ← Negated + possible + family history
├── ground_truth.json
└── README.md      ← this file
```

---

*Dataset version: 1.0.0 — Created 2026-09-19*
*For pipeline development and testing only. Not for clinical use.*
