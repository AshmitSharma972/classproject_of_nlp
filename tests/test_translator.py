"""Tests for the Hindi/Kannada translation gateway.

These tests mock network/offline engines: no external translation service is
needed to validate backend selection and actionable error reporting.
"""

from src.common import translator


def test_kannada_uses_google_translator_before_mymemory(monkeypatch):
    calls = []

    def google(text, language):
        calls.append(("google", language))
        return "The patient has fever"

    def mymemory(text, language):
        calls.append(("mymemory", language))
        return "should not be used"

    monkeypatch.setattr(translator, "_backend_deep_translator", google)
    monkeypatch.setattr(translator, "_backend_mymemory", mymemory)

    result = translator.translate_to_english("ರೋಗಿಗೆ ಜ್ವರ ಇದೆ", "Kannada (ಕನ್ನಡ)")

    assert result.was_translated is True
    assert result.translated_text == "The patient has fever"
    assert calls == [("google", "kn")]


def test_translation_network_error_is_actionable(monkeypatch):
    def unavailable(*_args):
        raise RuntimeError("NameResolutionError: getaddrinfo failed")

    monkeypatch.setattr(translator, "_backend_deep_translator", unavailable)
    monkeypatch.setattr(translator, "_backend_mymemory", unavailable)

    result = translator.translate_to_english("ರೋಗಿಗೆ ಜ್ವರ ಇದೆ", "Kannada (ಕನ್ನಡ)")

    assert result.was_translated is False
    assert "internet connection" in result.error.lower()
