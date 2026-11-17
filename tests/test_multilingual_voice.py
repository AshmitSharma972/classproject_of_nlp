"""Regression tests for local Hindi/Kannada/English microphone transcription."""
from __future__ import annotations

import http.client
import io
import json
import threading
import wave
from http.server import ThreadingHTTPServer

import pytest

import web_api
from src.common.speech_to_text import SpeechToTextError, _has_expected_script, _read_wav


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 80)
    return output.getvalue()


def test_wav_decoder_and_native_script_guard():
    samples, rate = _read_wav(_wav_bytes())
    assert rate == 16000 and samples.size == 80
    assert _has_expected_script("मुझे बुखार है", "Hindi")
    assert _has_expected_script("ನನಗೆ ಜ್ವರ ಇದೆ", "Kannada")
    assert not _has_expected_script("mujhe bukhar hai", "Hindi")
    assert not _has_expected_script("nanage jvara ide", "Kannada")
    with pytest.raises(SpeechToTextError):
        _read_wav(b"not a wav")


def test_transcribe_api_accepts_wav_and_language_field(monkeypatch):
    monkeypatch.setattr(web_api, "transcribe_audio", lambda audio, language: "ನನಗೆ ಜ್ವರ ಇದೆ")
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_api.ClinIQHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "ClinIQVoiceBoundary"
        audio = _wav_bytes()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\nKannada\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"microphone.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("POST", "/api/transcribe", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["text"] == "ನನಗೆ ಜ್ವರ ಇದೆ"
        assert payload["script"] == "Kannada"
        conn.close()
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_frontend_uses_local_multilingual_transcription():
    html = open("index.html", encoding="utf-8").read()
    assert "/api/transcribe" in html
    assert 'form.append("language", selectedLanguage())' in html
    assert "native script" in html
    assert 'voiceCard.classList.toggle("visible", !nonEnglish)' not in html
