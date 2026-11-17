"""Local HTTP bridge for the static ClinIQ frontend (127.0.0.1 only)."""
from __future__ import annotations

import hashlib
import json
import hmac
import logging
import os
import uuid
import secrets
import re
import zipfile
import io
from email.parser import BytesParser
from email.policy import default as email_policy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from dataclasses import asdict
from pydantic import ValidationError

from src.common.schemas import DemographicsInput
from src.common.translator import SUPPORTED_LANGUAGES, translate_from_english, translate_to_english
from src.common.speech_to_text import SpeechToTextError, transcribe_audio
from src.pipeline import ClinicalPipeline
from src.review.safety_flags import build_safety_flags
from src.review.audit_store import append_audit, verify_audit
from src.document_processor import DocumentProcessor
from src.utils import ScannedDocumentException
from src.summarization.templates import DISCLAIMER
from src.ner.lexicons import MEDICATION_LEXICON

HOST = "127.0.0.1"
PORT = 8765
MAX_TEXT_BYTES = 100 * 1024
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_REPORT_BYTES = 5 * 1024 * 1024
ALLOWED_ORIGINS = {o.strip() for o in os.getenv(
    "CLINIQ_CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
).split(",") if o.strip()}
pipeline = ClinicalPipeline()
logging.basicConfig(level=os.getenv("CLINIQ_LOG_LEVEL", "INFO"))
logger = logging.getLogger("cliniq.api")
_secret = os.getenv("CLINIQ_TRANSLATION_SECRET")
if not _secret:
    _secret = secrets.token_hex(32)
    logger.warning("CLINIQ_TRANSLATION_SECRET unset; generated an ephemeral translation-token secret")
_secret_bytes = _secret.encode("utf-8")
AUDIT_PATH = os.getenv("CLINIQ_AUDIT_PATH", "data/review_audit.jsonl")
document_processor = DocumentProcessor()


def _translation_token(language: str, original: str, english_text: str) -> str:
    message = language + "\x00" + original + "\x00" + english_text
    return hmac.new(_secret_bytes, message.encode("utf-8"), hashlib.sha256).hexdigest()


def _critical_token_report(original: str, english: str, back: str) -> dict:
    patterns = [
        ("number_or_unit", re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mmhg|°?[cf]|days?|weeks?|months?|years?|%)?\b", re.I)),
        ("negation", re.compile(r"\b(?:no|not|never|without|denies|cannot|లేదు|ಇಲ್ಲ)\b", re.I)),
        ("duration", re.compile(r"\b(?:for|since)\s+\w+\s*(?:days?|weeks?|months?|years?|hours?)?", re.I)),
    ]
    items = []
    for kind, pattern in patterns:
        for match in pattern.finditer(original):
            token = match.group(0).strip()
            en_ok = token.casefold() in english.casefold()
            back_ok = bool(back) and token.casefold() in back.casefold()
            state = "preserved" if en_ok and back_ok else "missing" if not en_ok or not back else "changed"
            items.append({"type": kind, "source_token": token, "preserved_in_english": en_ok,
                          "preserved_in_back_translation": back_ok, "status": state})
    for drug in sorted(MEDICATION_LEXICON, key=len, reverse=True):
        if re.search(rf"\b{re.escape(drug)}\b", original, re.I):
            en_ok = bool(re.search(rf"\b{re.escape(drug)}\b", english, re.I))
            back_ok = bool(back) and bool(re.search(rf"\b{re.escape(drug)}\b", back, re.I))
            state = "preserved" if en_ok and back_ok else "missing" if not en_ok or not back else "changed"
            items.append({"type": "drug_name", "source_token": drug, "preserved_in_english": en_ok,
                          "preserved_in_back_translation": back_ok, "status": state})
    risky = [i for i in items if i["status"] != "preserved"]
    risk = "high" if any(i["type"] in ("number_or_unit", "negation") and i["status"] == "missing" for i in risky) else "medium" if risky or not back else "low"
    return {"items": items, "risk_level": risk,
            "notice": "Heuristic token comparison only; not a calibrated confidence score."}


def _summary_items(record, source="english"):
    items = []
    index_by_key = {}
    for item in record.section_items:
        span = item.provenance_span
        evidence = [] if span is None else [{"start": span.start, "end": span.end, "source": source}]
        if evidence:
            # Keep each source mention as evidence, while returning one visible
            # summary fact for duplicate display text.
            key = " ".join((item.display_text or "").split()).casefold()
            if key in index_by_key:
                items[index_by_key[key]]["evidence"].extend(evidence)
                continue
            index_by_key[key] = len(items)
            items.append({"id": item.id, "text": item.display_text, "section": item.section.value,
                          "evidence": evidence, "unsupported": False})
    return items


def _create_docx_report(report: dict, demographics: dict) -> bytes:
    """Build a readable report in memory; no patient details are written to disk."""
    from docx import Document
    doc = Document()
    section = doc.sections[0]
    section.header.paragraphs[0].text = f"{demographics.get('full_name') or 'Not provided'}  |  Report {report.get('report_id') or 'Not provided'}"
    footer = section.footer.paragraphs[0]
    footer.text = "Decision-support summary only. Not a diagnosis or treatment recommendation. Must be reviewed by a licensed physician."
    doc.add_heading("Clinical Information Summary", 0)
    doc.add_paragraph(f"Report ID: {report.get('report_id') or 'Not provided'}")
    doc.add_paragraph(f"Generated: {report.get('generated_at') or 'Not provided'}")
    doc.add_paragraph(f"Visit date: {demographics.get('visit_date') or 'Not provided'}")
    doc.add_paragraph(f"Clinic / department: {demographics.get('department') or 'Not provided'}")

    def value(obj, key):
        val = obj.get(key)
        return "Not provided" if val in (None, "") else str(val)

    doc.add_heading("Patient information", level=1)
    height = demographics.get("height_cm"); weight = demographics.get("weight_kg")
    bmi = f"{float(weight) / (float(height) / 100) ** 2:.1f} (calculated)" if weight and height else "Not provided"
    rows = [
        ("Full name", value(demographics, "full_name")),
        ("Age / sex", f"{value(demographics, 'age')} / {value(demographics, 'sex')}"),
        ("Patient ID", value(demographics, "patient_id")), ("Phone", value(demographics, "phone")),
        ("Address / city", value(demographics, "address_city")), ("Weight", value(demographics, "weight_kg") + (" kg" if weight else "")),
        ("Height", value(demographics, "height_cm") + (" cm" if height else "")), ("BMI", bmi),
        ("Blood group", value(demographics, "blood_group")),
        ("Emergency contact", f"{value(demographics, 'emergency_contact_name')} / {value(demographics, 'emergency_contact_phone')}"),
        ("Referring physician", value(demographics, "referring_physician")),
    ]
    table = doc.add_table(rows=0, cols=2); table.style = "Light Shading Accent 1"
    for label, val in rows:
        cells = table.add_row().cells; cells[0].text = label; cells[1].text = val

    def unique_values(values):
        seen, result = set(), []
        for value in values:
            text = str(value or "").strip()
            key = " ".join(text.split()).casefold()
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result

    patient = report.get("patient_record") or {}
    sections = patient.get("sections") or {}
    entities = patient.get("entities", [])
    def add_lines(title, items):
        doc.add_heading(title, level=1)
        if items:
            for item in items:
                doc.add_paragraph(str(item), style="List Bullet")
        else:
            doc.add_paragraph("Not provided")

    doc.add_heading("Chief complaints", level=1)
    complaint_table = doc.add_table(rows=1, cols=3); complaint_table.style = "Light Shading Accent 1"
    for cell, label in zip(complaint_table.rows[0].cells, ("Complaint", "Duration", "Assertion status")): cell.text = label
    assertions = {item.get("entity_id"): item.get("status") for item in patient.get("assertions", [])}
    time_expressions = patient.get("time_expressions", [])
    complaint_entities = []
    complaint_keys = set()
    for item in entities:
        if item.get("type") not in {"SYMPTOM", "DISEASE"}:
            continue
        key = (item.get("type"), " ".join(str((item.get("span") or {}).get("text", "")).split()).casefold())
        if key not in complaint_keys:
            complaint_keys.add(key)
            complaint_entities.append(item)
    for entity in complaint_entities:
        linked_time = next((time for time in time_expressions if entity.get("id") in time.get("linked_entity_ids", [])), None)
        status = assertions.get(entity.get("id"), "Unreviewed")
        status = {"PRESENT": "Present", "NEGATED": "Absent"}.get(status, status.title() if status else "Unreviewed")
        cells = complaint_table.add_row().cells
        cells[0].text = (entity.get("span") or {}).get("text", "Not provided")
        cells[1].text = (linked_time or {}).get("raw_value") or (linked_time or {}).get("normalized_value") or "Not provided"
        cells[2].text = status
    if not complaint_entities:
        cells = complaint_table.add_row().cells
        for cell in cells: cell.text = "Not provided"
    add_lines("History of present illness", [report.get("summary") or "Insufficient data"])
    add_lines("Past medical history", sections.get("Past History", []))
    add_lines("Family history", sections.get("Family History", []))
    doc.add_heading("Current medications", level=1)
    med_table = doc.add_table(rows=1, cols=4); med_table.style = "Light Shading Accent 1"
    for cell, label in zip(med_table.rows[0].cells, ("Drug", "Dose", "Frequency", "Source")): cell.text = label
    entity_by_id = {entity.get("id"): entity for entity in entities}
    relations = patient.get("relations", [])
    narration_meds = []
    medication_keys = set()
    for entity in entities:
        if entity.get("type") != "MEDICATION":
            continue
        key = " ".join(str((entity.get("span") or {}).get("text", "")).split()).casefold()
        if key not in medication_keys:
            medication_keys.add(key)
            narration_meds.append(entity)
    for medication in narration_meds:
        related = [relation for relation in relations if relation.get("head_entity_id") == medication.get("id")]
        def relation_text(kind):
            link = next((rel for rel in related if rel.get("relation_type") == kind), None)
            target = entity_by_id.get(link.get("tail_entity_id")) if link else None
            return (target.get("span") or {}).get("text") if target else "Not provided"
        cells = med_table.add_row().cells
        cells[0].text = (medication.get("span") or {}).get("text", "Not provided")
        cells[1].text = relation_text("MEDICATION-DOSAGE")
        cells[2].text = relation_text("MEDICATION-FREQUENCY")
        cells[3].text = "Narration"
    for item in sections.get("Drug History", []):
        if not narration_meds:
            cells = med_table.add_row().cells; cells[0].text = str(item); cells[1].text = "Not provided"; cells[2].text = "Not provided"; cells[3].text = "Narration"
    if demographics.get("current_medications"):
        for medication in str(demographics["current_medications"]).splitlines():
            cells = med_table.add_row().cells; cells[0].text = medication; cells[1].text = "Not provided"; cells[2].text = "Not provided"; cells[3].text = "Patient form"
    if len(med_table.rows) == 1:
        cells = med_table.add_row().cells
        for cell in cells: cell.text = "Not provided"
    add_lines("Allergies from narration", sections.get("Allergy", []))
    add_lines("Allergies from patient form", [value(demographics, "known_allergies")])
    add_lines("Vitals and test results", unique_values([e.get("span", {}).get("text", "") for e in patient.get("entities", []) if e.get("type") in {"TEST", "TEST_RESULT"}]))
    add_lines("Timeline", [entry.get("label", "") for entry in patient.get("timeline", [])])
    add_lines("Safety and completeness flags", [f.get("message", "") for f in report.get("flags", [])])
    add_lines("Pipeline provenance", [f"Pipeline output · request {report.get('request_id') or 'Not provided'} · language {report.get('language') or 'Not provided'} · translation risk {report.get('translation_risk_level') or 'Not provided'}"])
    add_lines("Physician review status", report.get("review_status") or ["Not reviewed"])
    add_lines("Audit chain status", [report.get("audit_chain_status") or "Not verified"])
    doc.add_heading("Sign-off", level=1)
    doc.add_paragraph(f"Examining physician: {value(demographics, 'examining_physician')}\nSignature: ______________________________\nDate/time: ______________________________")
    doc.add_heading("Patient narration — original", level=1); doc.add_paragraph(report.get("original_text") or "Not provided")
    doc.add_heading("Patient narration — English", level=1); doc.add_paragraph(report.get("english_text") or "Not provided")
    doc.add_heading("Required disclaimer", level=1)
    doc.add_paragraph("Decision-support summary only. Not a diagnosis or treatment recommendation. Must be reviewed by a licensed physician.")
    stream = io.BytesIO(); doc.save(stream); return stream.getvalue()


class ClinIQHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict[str, Any], request_id: str | None = None) -> None:
        if request_id:
            payload.setdefault("request_id", request_id)
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_binary(self, payload: bytes, filename: str, request_id: str) -> None:
        self.send_response(200)
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("X-Request-ID", request_id)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self, limit: int = MAX_TEXT_BYTES) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        if not raw_length.isdigit():
            raise ValueError("Invalid Content-Length")
        length = int(raw_length)
        if length > limit:
            raise OverflowError("Request body exceeds configured limit")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    def _read_upload(self, limit: int = MAX_UPLOAD_BYTES) -> tuple[str, bytes, dict[str, str]]:
        raw_length = self.headers.get("Content-Length", "0")
        if not raw_length.isdigit() or int(raw_length) > limit + 64 * 1024:
            raise OverflowError("Upload exceeds configured limit")
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart/form-data")
        message = BytesParser(policy=email_policy).parsebytes(
            (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + self.rfile.read(int(raw_length)))
        )
        fields: dict[str, str] = {}
        for part in message.iter_parts():
            field_name = part.get_param("name", header="content-disposition")
            if field_name and field_name != "file" and not part.get_filename():
                fields[field_name] = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
                continue
            if part.get_param("name", header="content-disposition") == "file":
                name = part.get_filename() or "upload"
                data = part.get_payload(decode=True) or b""
                if len(data) > limit:
                    raise OverflowError("Upload exceeds configured limit")
                return name, data, fields
        raise ValueError("A file field is required")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        request_id = str(uuid.uuid4())
        if self.path == "/api/benchmark":
            try:
                from evaluation.run_eval import run as run_benchmark
                self._send(200, run_benchmark(), request_id)
            except Exception:
                logger.exception("request_id=%s benchmark evaluation failed", request_id)
                self._send(500, {"error": "Benchmark evaluation failed"}, request_id)
        elif self.path == "/api/review/verify":
            self._send(200, verify_audit(AUDIT_PATH), request_id)
        elif self.path == "/api/review":
            try:
                with open(AUDIT_PATH, encoding="utf-8") as stream:
                    actions = [json.loads(line) for line in stream if line.strip()]
            except FileNotFoundError:
                actions = []
            self._send(200, {"actions": actions}, request_id)
        else:
            self._send(404, {"error": "Unknown endpoint"}, request_id)

    def do_POST(self) -> None:  # noqa: N802
        request_id = str(uuid.uuid4())
        try:
            if self.path == "/api/transcribe":
                filename, audio, fields = self._read_upload(MAX_AUDIO_BYTES)
                language = fields.get("language", "")
                if language not in {"English", "Hindi", "Kannada"}:
                    self._send(400, {"error": "language must be English, Hindi, or Kannada"}, request_id)
                    return
                if not filename.lower().endswith(".wav") or not (audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"):
                    self._send(400, {"error": "A WAV microphone recording is required"}, request_id)
                    return
                transcript = transcribe_audio(audio, language)
                self._send(200, {
                    "text": transcript,
                    "language": language,
                    "script": {"English": "Latin", "Hindi": "Devanagari", "Kannada": "Kannada"}[language],
                    "engine": "local-whisper",
                }, request_id)
                return
            if self.path == "/api/document":
                filename, data, _fields = self._read_upload()
                ext = os.path.splitext(filename)[1].lower()
                if ext not in {".pdf", ".docx", ".txt"}:
                    self._send(400, {"error": "Only PDF, DOCX and TXT files are supported"}, request_id)
                    return
                if (ext == ".pdf" and not data.startswith(b"%PDF-")) or (ext == ".docx" and not data.startswith(b"PK\x03\x04")) or (ext == ".txt" and b"\x00" in data):
                    self._send(400, {"error": "File extension and file contents do not match"}, request_id)
                    return
                if ext == ".docx":
                    try:
                        with zipfile.ZipFile(io.BytesIO(data)) as archive:
                            if "word/document.xml" not in archive.namelist():
                                raise ValueError("DOCX content is missing document.xml")
                    except Exception:
                        self._send(400, {"error": "Invalid DOCX file"}, request_id)
                        return
                try:
                    extracted = document_processor.extract_from_bytes(data, filename)
                except ScannedDocumentException:
                    self._send(422, {"error": "Scanned PDF detected; OCR is not available"}, request_id)
                    return
                if ext == ".pdf" and extracted.get("is_scanned"):
                    self._send(422, {"error": "Scanned PDF detected; OCR is not available"}, request_id)
                    return
                document_text = str(extracted.get("text", "")).strip()
                if not document_text:
                    self._send(422, {"error": "Document contains no extractable text"}, request_id)
                    return
                result = pipeline.run(document_text)
                page_refs = []
                current_page = None
                for i, match in enumerate(re.finditer(r"[^\n]+", document_text)):
                    page_marker = re.match(r"---\s*Page\s+(\d+)\s*---", match.group(0), re.I)
                    if page_marker:
                        current_page = int(page_marker.group(1))
                        continue
                    if match.group(0).strip():
                        page_refs.append({"page": current_page, "paragraph": len(page_refs) + 1,
                                          "start": match.start(), "end": match.end()})
                self._send(200, {"source_name": filename, "file_type": extracted.get("file_type"),
                                 "page_count": extracted.get("page_count"), "summary": result.summary,
                                 "generated_at": datetime.now(timezone.utc).isoformat(),
                                 "pipeline": result.model_dump(mode="json"), "patient_record": result.patient_record.model_dump(mode="json"),
                                 "summary_items": _summary_items(result.patient_record),
                                 "disclaimer": DISCLAIMER.strip(),
                                 "evidence_references": page_refs,
                                 "flags": build_safety_flags(document_text, result.entities, result.assertions, result.time_expressions, result.relations)}, request_id)
                return
            request = self._read_json(MAX_REPORT_BYTES if self.path == "/api/report/docx" else MAX_TEXT_BYTES)
            if self.path == "/api/report/docx":
                demo_input = request.get("demographics")
                if not isinstance(demo_input, dict):
                    self._send(400, {"error": "demographics is required"}, request_id)
                    return
                demographics = DemographicsInput.model_validate(demo_input).to_patient_demographics()
                values = asdict(demographics)
                values["patient_id"] = values.get("patient_id") or f"CLQ-{secrets.token_hex(3).upper()}"
                document = _create_docx_report(request.get("report") or {}, values)
                self._send_binary(document, "cliniq-clinical-report.docx", request_id)
                return
            if self.path == "/api/translate/confirm":
                original = request.get("original")
                language = request.get("language")
                english = request.get("english_text")
                reviewer = request.get("reviewer")
                if not all(isinstance(x, str) and x.strip() for x in (original, language, english, reviewer)):
                    self._send(400, {"error": "original, language, english_text and reviewer are required"}, request_id)
                    return
                if language not in SUPPORTED_LANGUAGES or language == "English":
                    self._send(400, {"error": "A supported non-English language is required"}, request_id)
                    return
                self._send(200, {"translation_token": _translation_token(language, original, english)}, request_id)
                return
            if self.path == "/api/review":
                action = request.get("action")
                item_id = request.get("item_id")
                reviewer = request.get("reviewer")
                if action not in {"accept", "edit", "reject", "verify"} or not all(isinstance(v, str) and v.strip() for v in (item_id, reviewer)):
                    self._send(400, {"error": "action, item_id and reviewer are required"}, request_id)
                    return
                if action == "edit" and not isinstance(request.get("new_value"), str):
                    self._send(400, {"error": "new_value is required for edit"}, request_id)
                    return
                payload = {"id": str(uuid.uuid4()), "action": action, "item_id": item_id,
                           "reviewer": reviewer.strip(), "comment": request.get("comment"),
                           "new_value": request.get("new_value"),
                           "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id}
                entry = append_audit(AUDIT_PATH, payload)
                self._send(200, {"review": entry, "saved": True}, request_id)
                return
            text = request.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send(400, {"error": "A non-empty text string is required"}, request_id)
                return
            if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                self._send(413, {"error": "Text exceeds the 100 KB limit"}, request_id)
                return
            language = request.get("language", "English")
            if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
                self._send(400, {"error": "Unsupported language"}, request_id)
                return
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            logger.info("request_id=%s path=%s text_length=%d text_sha256=%s",
                        request_id, self.path, len(text), digest)

            if self.path == "/api/translate":
                result = translate_to_english(text, language)
                if result.error:
                    # translator.error contains a user-safe, non-input diagnostic.
                    self._send(502, {"error": result.error}, request_id)
                    return
                back = translate_from_english(result.translated_text, result.source_language_code) if result.was_translated else ""
                report = _critical_token_report(text, result.translated_text, back)
                self._send(200, {
                    "original_text": result.original_text,
                    "translated_text": result.translated_text,
                    "english": result.translated_text,
                    "original": result.original_text,
                    "source_language": result.source_language,
                    "source_language_code": result.source_language_code,
                    "was_translated": result.was_translated,
                    "translated": result.was_translated,
                    "language_code": result.source_language_code,
                    "back_translation": back,
                    "critical_token_report": report,
                    "translation_token": _translation_token(language, text, result.translated_text) if result.was_translated else None,
                    "disclaimer": DISCLAIMER.strip(),
                }, request_id)
                return
            if self.path == "/api/analyze":
                demographics_input = request.get("demographics")
                if not isinstance(demographics_input, dict):
                    self._send(400, {"error": "Valid patient demographics and consent are required"}, request_id)
                    return
                demographics_model = DemographicsInput.model_validate(demographics_input)
                demographics = demographics_model.to_patient_demographics()
                demographics.patient_id = demographics.patient_id or f"CLQ-{secrets.token_hex(3).upper()}"
                demographics_payload = asdict(demographics)
                demographics_payload["consent"] = demographics_model.consent
                pipeline_text = text
                if language != "English":
                    english_text = request.get("english_text")
                    token = request.get("translation_token")
                    if not isinstance(english_text, str) or not isinstance(token, str):
                        self._send(400, {"error": "english_text and translation_token are required"}, request_id)
                        return
                    expected = _translation_token(language, text, english_text)
                    if not hmac.compare_digest(token, expected):
                        self._send(400, {"error": "Translation confirmation token is invalid or stale"}, request_id)
                        return
                    pipeline_text = english_text
                result = pipeline.run(pipeline_text)
                flags = build_safety_flags(pipeline_text, result.entities, result.assertions,
                                           result.time_expressions, result.relations,
                                           demographics=demographics_payload)
                self._send(200, {
                    "patient_record": result.patient_record.model_dump(mode="json"),
                    "demographics": demographics_payload,
                    "summary": result.summary,
                    "original_text": text,
                    "english_text": pipeline_text,
                    "summary_items": _summary_items(result.patient_record),
                    "disclaimer": DISCLAIMER.strip(),
                    "pipeline": result.model_dump(mode="json"),
                    "flags": flags,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }, request_id)
                return
            self._send(404, {"error": "Unknown endpoint"}, request_id)
        except OverflowError:
            self._send(413, {"error": "Request body exceeds the configured size limit"}, request_id)
        except SpeechToTextError as exc:
            self._send(422, {"error": str(exc)}, request_id)
        except ValidationError as exc:
            field_errors = [".".join(str(p) for p in error["loc"]) + ": " + error["msg"]
                            for error in exc.errors()]
            self._send(400, {"error": "Invalid demographics or report input", "details": field_errors}, request_id)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": "Invalid request body"}, request_id)
        except Exception:
            logger.exception("request_id=%s unhandled API error", request_id)
            self._send(500, {"error": "Internal server error"}, request_id)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("http_access client=%s %s", self.client_address[0], format % args)


if __name__ == "__main__":
    logger.info("ClinIQ API listening at http://%s:%d", HOST, PORT)
    ThreadingHTTPServer((HOST, PORT), ClinIQHandler).serve_forever()
