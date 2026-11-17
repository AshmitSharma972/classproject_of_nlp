# Clinician review queue

`clinical_cases.jsonl` is the initial synthetic review queue: 30 English, 15 Hindi, and 15 Kannada narrations. Every row is explicitly marked `TODO: clinician annotation and approval`; there are no clinician-approved annotations yet. The `gold` name is a directory convention, not a claim that these draft labels or translations are validated.

Before using a row for metrics, a qualified clinician should fill `english_text` for Hindi/Kannada, add `annotations` (`entities`, `assertions`, `temporal_links`, `medication_relations`), and set `review_status` to `clinician_approved`. Do not remove the synthetic-data limitation from reports.
