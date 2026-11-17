"""Regression checks for the local translation and audit trust layer."""
import json
from pathlib import Path
import threading
import http.client
import json as jsonlib
from http.server import ThreadingHTTPServer

import web_api
from web_api import _critical_token_report, _translation_token
from src.review.audit_store import append_audit, verify_audit


def test_english_translation_token_is_bound_to_source_and_translation():
    token = _translation_token("Hindi", "मुझे बुखार है", "I have fever")
    assert token == _translation_token("Hindi", "मुझे बुखार है", "I have fever")
    assert token != _translation_token("Hindi", "मुझे खांसी है", "I have fever")
    assert token != _translation_token("Hindi", "मुझे बुखार है", "I do not have fever")


def test_critical_token_report_is_explicitly_heuristic_and_flags_missing_negation():
    report = _critical_token_report("No fever for 5 days", "Fever for 5 days", "")
    assert "not a calibrated confidence score" in report["notice"]
    assert report["risk_level"] == "high"
    assert any(item["type"] == "negation" and item["status"] != "preserved" for item in report["items"])


def test_review_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "review.jsonl"
    append_audit(str(path), {"id": "a1", "action": "accept", "reviewer": "Reviewer"})
    append_audit(str(path), {"id": "a2", "action": "verify", "reviewer": "Reviewer"})
    assert verify_audit(str(path))["valid"]
    lines = path.read_text(encoding="utf-8").splitlines()
    item = json.loads(lines[0]); item["reviewer"] = "Tampered"
    lines[0] = json.dumps(item)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_audit(str(path))["first_broken_index"] == 0


def test_english_fixture_has_at_least_40_cases():
    fixture = Path(__file__).parent / "fixtures" / "english_narrations.txt"
    cases = [line for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cases) >= 40


def test_api_trust_review_and_document_endpoints(tmp_path, caplog):
    old_audit = web_api.AUDIT_PATH
    web_api.AUDIT_PATH = str(tmp_path / "audit.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_api.ClinIQHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        demographics = {"full_name": "Diabetes Rao Private", "age": 34, "sex": "Not stated", "consent": True,
                       "address_city": "LOG-SECRET-ADDRESS"}
        body = jsonlib.dumps({"text": "मुझे बुखार है", "language": "Hindi", "demographics": demographics,
                              "english_text": "I have fever", "translation_token": "invalid"})
        conn.request("POST", "/api/analyze", body, {"Content-Type": "application/json", "Origin": "http://127.0.0.1:5500"})
        response = conn.getresponse(); payload = jsonlib.loads(response.read())
        assert response.status == 400
        assert payload["request_id"]
        assert response.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:5500"
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        confirm = jsonlib.dumps({"original": "मुझे बुखार है", "language": "Hindi",
                                 "english_text": "I have fever", "reviewer": "QA Reviewer"})
        conn.request("POST", "/api/translate/confirm", confirm, {"Content-Type": "application/json"})
        response = conn.getresponse(); confirmed = jsonlib.loads(response.read())
        assert response.status == 200 and confirmed["translation_token"]
        token = confirmed["translation_token"]
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        caplog.set_level("INFO", logger="cliniq.api")
        body = jsonlib.dumps({"text": "मुझे बुखार है", "language": "Hindi", "demographics": demographics,
                              "english_text": "I have fever", "translation_token": token})
        conn.request("POST", "/api/analyze", body, {"Content-Type": "application/json"})
        response = conn.getresponse(); payload = jsonlib.loads(response.read())
        assert response.status == 200
        assert payload["original_text"] == "मुझे बुखार है"
        assert payload["english_text"] == "I have fever"
        assert payload["pipeline"]["raw_text"] == "I have fever"
        assert "Decision support only" in payload["disclaimer"]
        assert payload["demographics"]["patient_id"].startswith("CLQ-")
        assert "consent" not in payload["patient_record"]
        assert not any("Diabetes" in entity["span"]["text"] for entity in payload["patient_record"]["entities"])
        assert "Diabetes Rao Private" not in caplog.text and "LOG-SECRET-ADDRESS" not in caplog.text
        conn.close()

        demographics["full_name"] = "Fever Kumar"
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        body = jsonlib.dumps({"text": "I have a cough.", "language": "English", "demographics": demographics})
        conn.request("POST", "/api/analyze", body, {"Content-Type": "application/json"})
        response = conn.getresponse(); fever_name_result = jsonlib.loads(response.read())
        assert response.status == 200
        assert not any(entity["span"]["text"].casefold() == "fever" for entity in fever_name_result["patient_record"]["entities"])
        assert "Fever Kumar" not in caplog.text
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        export = jsonlib.dumps({"demographics": payload["demographics"], "report": payload})
        conn.request("POST", "/api/report/docx", export, {"Content-Type": "application/json"})
        response = conn.getresponse(); docx = response.read()
        assert response.status == 200 and docx[:2] == b"PK"
        assert "word/document.xml" in __import__("zipfile").ZipFile(__import__("io").BytesIO(docx)).namelist()
        conn.close()

        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        review = jsonlib.dumps({"action": "accept", "item_id": "demo-item", "reviewer": "QA Reviewer"})
        conn.request("POST", "/api/review", review, {"Content-Type": "application/json"})
        response = conn.getresponse(); assert response.status == 200; response.read(); conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/api/review/verify")
        response = conn.getresponse(); assert jsonlib.loads(response.read())["valid"]; conn.close()

        boundary = "ClinIQTestBoundary"
        file_bytes = b"I have cough for 3 days."
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"case.txt\"\r\n"
                "Content-Type: text/plain\r\n\r\n").encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("POST", "/api/document", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        response = conn.getresponse(); doc_result = jsonlib.loads(response.read())
        assert response.status == 200 and doc_result["page_count"] == 1
        assert "Decision support only" in doc_result["disclaimer"]
        conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
        web_api.AUDIT_PATH = old_audit
