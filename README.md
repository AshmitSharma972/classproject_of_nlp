# Patient Case-Taking Software using NLP(ClinIQ)

ClinIQ is a local, multilingual clinical information-assistance prototype. It extracts and organizes reported information for physician review. **It is not a diagnostic tool and does not diagnose or recommend treatment.** Every generated report must retain the decision-support disclaimer and be reviewed by a qualified clinician.

## Current flow

The browser UI is `index.html`; `web_api.py` exposes a loopback-only Python API on `127.0.0.1:8765`. The API sends English text through the nine-stage `ClinicalPipeline`: preprocessing, NER, assertion/negation, temporal extraction, relation extraction, section classification, PatientRecord assembly, template summary, and physician-review readiness. Hindi and Kannada inputs pass through translation review before the English pipeline. The active summary is the deterministic template in `src/summarization/summarizer.py`; `src/summarizer.py` is legacy.

## Run locally (Windows)

Install dependencies in the project's Python environment, then start two terminals:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Terminal 1:

```powershell
.\run_web_api.bat
```

Terminal 2:

```powershell
.\run_frontend.bat
```

Open `http://127.0.0.1:5500/`. Serving the page from this origin is required for the default CORS policy. `run_server.bat` is a legacy Streamlit launcher.

## API

All requests are local to the host unless translation is enabled (see Privacy).

| Method and path | Purpose |
|---|---|
| `POST /api/translate` | Translate source text; returns source/English/back-translation, heuristic critical-token report, and translation token. |
| `POST /api/translate/confirm` | Confirm the exact source and user-edited English text; returns an HMAC token. |
| `POST /api/analyze` | Analyze English text directly. Hindi/Kannada requests require `english_text` and a valid `translation_token`. |
| `POST /api/document` | Multipart upload of PDF, DOCX or TXT (10 MB maximum); extracts text and runs the pipeline. |
| `GET /api/benchmark` | Return per-language clinician-approved pipeline metrics, local summary-checkpoint status, saved training metrics, and seeded translation checks for the Benchmark page. |
| `POST /api/review` | Append accept/edit/reject/verify review action. Requires reviewer name and item ID. |
| `GET /api/review` | Read the append-only review action list. |
| `GET /api/review/verify` | Recompute the tamper-evident audit chain. |

JSON narration requests are limited to 100 KB. Upload signature and extension are checked. Scanned PDFs are rejected with an OCR-unavailable message. API errors contain a generic message and request ID; details remain in server logs. Logs include text length and a short hash, never the narration.

## Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `CLINIQ_CORS_ORIGINS` | `http://127.0.0.1:5500,http://localhost:5500` | Comma-separated allowed browser origins. |
| `CLINIQ_TRANSLATION_SECRET` | Random per process, warning logged | HMAC key for translation confirmations. Set a stable random secret for persistent tokens across API restarts. |
| `CLINIQ_TRANSLATION_OFFLINE` | `0` | Set to `1` to disable online translation providers. Hindi translation requires a locally installed Argos model; Kannada may be unavailable offline. |
| `CLINIQ_ASR_MODEL` | `openai/whisper-base` | Local multilingual speech-to-text model used by `/api/transcribe`. It downloads on first use, then is cached locally. |
| `CLINIQ_AUDIT_PATH` | `data/review_audit.jsonl` | Append-only local review audit file. |
| `CLINIQ_LOG_LEVEL` | `INFO` | Python logging level. |

Translation currently uses external providers when offline mode is disabled. Source narration and English translation are sent to that provider. Back-translation and critical-token checks are heuristic comparisons, **not calibrated confidence or clinical validation**. The UI displays this notice and requires explicit user confirmation; any translation edit invalidates the prior token.

Microphone input records browser audio and sends it only to the loopback API. The API uses the local multilingual Whisper model in transcription mode for English, Hindi, and Kannada. Hindi output is checked for Devanagari text and Kannada output for Kannada text; text returned in Romanized Hinglish or Kanglish is rejected instead of being silently used. The first microphone transcription may download the configured model weights.

## Tests and evaluation

Run `python -m pytest -q`. `tests/fixtures/english_narrations.txt` contains synthetic English cases. `evaluation/gold/clinical_cases.jsonl` has 30 English, 15 Hindi, and 15 Kannada synthetic clinician-review slots. All are currently TODO and excluded from accuracy metrics. Fill/approve annotations with a clinician, then run `python evaluation/run_eval.py`; it reports entity P/R/F1, assertion accuracy, temporal-link accuracy, medication-relation accuracy, and seeded translation-token checks by language. Synthetic cases do not prove clinical performance.

The Benchmark page calls `GET /api/benchmark`. It distinguishes **template-based live summary output** from the separately trained mT5 summarizer. `scripts/train_multilingual_summarizer.py` saves test metrics and training metadata to `training_metrics.json` inside the model output directory. The page may show training loss or other recorded trainer statistics, but labels these as training statistics, not clinical accuracy. Clinical pipeline accuracy stays unavailable until cases have clinician-approved annotations. Existing weights without a metrics file are identified as weights with unverified fine-tuning status; their accuracy cannot be reconstructed from the weight file alone.

To evaluate an existing local summary checkpoint without retraining, run `python scripts/evaluate_summarizer_checkpoint.py --model models/clinical_transformer_weights --max-samples 100`. It generates the held-out synthetic cases and saves ROUGE text-overlap scores beside the model as `summary_evaluation.json`; refresh the Benchmark page to display them. ROUGE is not an accuracy or clinical-validity score.

## Known limitations

- NER, assertion, temporal, and relation extraction use deterministic lexicons and rules; coverage is incomplete, and ambiguous language needs physician verification.
- Hindi/Kannada translation may lose clinical meaning. Token matching detects only a limited set of numbers, units, negation terms, and duration patterns; it can miss errors.
- Translation offsets are not aligned to original-script text. Source highlighting for translated input must not claim exact original-language offsets until an alignment method is implemented.
- Document references are paragraph-level approximations; OCR and embedded images are not processed.
- The audit chain detects edits to the local JSONL file but is not backed by a trusted server. Reviewer names are not authenticated; real authentication and protected storage are future work.
- Safety flags are completeness prompts and heuristics, not diagnosis or treatment advice.
- No synthetic benchmark establishes clinical validity. Human clinician review and a prospectively designed evaluation are required before any clinical use.
- The initial multilingual evaluation rows are review slots only; no clinically reviewed gold annotations or per-language pipeline accuracy results are available yet.

## Handling patient data

Patient demographics and narration are held in browser memory and sent to the local API only when the user analyzes a case or explicitly exports a report. The API does not save demographics or include them in request logs. Review actions are persisted separately in the configured append-only audit JSONL file; do not put patient details in reviewer comments. DOCX/PDF exports contain identifiable information, so store and share them only under your institution's policy. The Clear all patient data control clears the form, narration, rendered report, and browser session storage.

ClinIQ is a prototype. Use synthetic or properly consented data only. It is not compliant with HIPAA, GDPR, or any other medical-privacy regulation by itself, and the local API, browser, operating system, and exported files do not provide production-grade access controls or encrypted clinical storage. Do not use it as a system of record.

## Safety statement

**Decision support only. Not a diagnosis or treatment recommendation. The physician remains responsible for clinical decisions.**
# classproject_of_nlp
