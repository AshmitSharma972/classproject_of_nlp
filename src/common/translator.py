"""
Translation utility module for the Clinical Case-Taking Pipeline.

Backend priority:
  1. argostranslate  -- offline, no rate limits, primary for Hindi
  2. deep-translator / Google Translate -- primary network translator
  3. deep-translator / MyMemory -- network fallback

Constraints:
- Pipeline operates on English text ONLY.
- Translation is a UI/pre-processing concern; pipeline modules are not modified.
- Provenance: original text and source language always preserved.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported languages for TEXT INPUT (voice is English-only)
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Hindi (हिंदी)": "hi",
    "Hindi": "hi",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Kannada": "kn",
}

_LANG_DISPLAY_TO_CODE: dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
}

_SCRIPT_RANGES: dict[str, tuple[str, str]] = {
    "hi": ("ऀ", "ॿ"),
    "kn": ("ಀ", "೿"),
}

_ARGOS_SUPPORTED: frozenset[str] = frozenset({"hi"})


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_language: str
    source_language_code: str
    target_language_code: str = "en"
    was_translated: bool = False
    error: Optional[str] = None


def translate_to_english(text: str, source_language: str) -> "TranslationResult":
    lang_code = SUPPORTED_LANGUAGES.get(source_language)

    if lang_code is None:
        lower = source_language.lower()
        for key, code in _LANG_DISPLAY_TO_CODE.items():
            if lower.startswith(key):
                lang_code = code
                break

    if lang_code is None:
        lang_code = _detect_lang_from_script(text) or "en"

    if lang_code == "en":
        logger.info("Input language is English -- translation skipped.")
        return TranslationResult(
            original_text=text, translated_text=text,
            source_language=source_language, source_language_code="en",
            was_translated=False,
        )

    logger.info("Translating %s (%s) -> English | %d chars", source_language, lang_code, len(text))

    backends = []
    offline = os.getenv("CLINIQ_TRANSLATION_OFFLINE", "0").lower() in {"1", "true", "yes"}
    if lang_code in _ARGOS_SUPPORTED:
        backends.append(("argostranslate", lambda lc=lang_code: _backend_argostranslate(text, lc)))
    if not offline:
        backends.append(("deep-translator",  lambda lc=lang_code: _backend_deep_translator(text, lc)))
        backends.append(("mymemory",         lambda lc=lang_code: _backend_mymemory(text, lc)))

    last_error = "No backend available"
    errors: list[str] = []

    for name, backend_fn in backends:
        try:
            logger.info("Trying backend: %s", name)
            candidate = backend_fn()
        except Exception as exc:
            last_error = str(exc)
            errors.append(last_error)
            logger.warning("Backend '%s' failed (%s)", name, type(exc).__name__)
            continue

        if not candidate or not candidate.strip():
            last_error = f"Backend '{name}' returned empty result"
            logger.warning(last_error); continue

        if _contains_source_script(candidate, lang_code):
            last_error = f"Backend '{name}' returned untranslated text"
            logger.warning(last_error); continue

        if candidate.strip() == text.strip():
            last_error = f"Backend '{name}' returned text identical to input"
            logger.warning(last_error); continue

        logger.info("Translation succeeded via '%s' | %d -> %d chars", name, len(text), len(candidate))
        return TranslationResult(
            original_text=text, translated_text=candidate,
            source_language=source_language, source_language_code=lang_code,
            was_translated=True,
        )

    friendly = _friendly_error(" | ".join(errors) or last_error)
    logger.error("All translation backends failed. Last error: %s", last_error)
    return TranslationResult(
        original_text=text, translated_text=text,
        source_language=source_language, source_language_code=lang_code,
        was_translated=False, error=friendly,
    )


def translate_from_english(text: str, target_language_code: str) -> str:
    """Translate English back to a supported language; returns empty on failure."""
    if os.getenv("CLINIQ_TRANSLATION_OFFLINE", "0").lower() in {"1", "true", "yes"}:
        return ""
    try:
        from deep_translator import GoogleTranslator  # type: ignore
        target = {"hi": "hi", "kn": "kn"}.get(target_language_code)
        return GoogleTranslator(source="en", target=target).translate(text) or "" if target else ""
    except Exception as exc:
        logger.warning("Back translation failed (%s)", type(exc).__name__)
        return ""


def _detect_lang_from_script(text: str) -> Optional[str]:
    for lang_code, (lo, hi) in _SCRIPT_RANGES.items():
        if any(lo <= ch <= hi for ch in text):
            return lang_code
    return None


def _contains_source_script(text: str, lang_code: str) -> bool:
    rng = _SCRIPT_RANGES.get(lang_code)
    if not rng:
        return False
    lo, hi = rng
    return any(lo <= ch <= hi for ch in text)


def _backend_argostranslate(text: str, source_lang: str) -> str:
    from argostranslate import translate  # type: ignore
    installed = translate.get_installed_languages()
    src = next((l for l in installed if l.code == source_lang), None)
    tgt = next((l for l in installed if l.code == "en"), None)
    if not src or not tgt:
        raise RuntimeError(f"argostranslate: {source_lang}->en package not installed")
    t = src.get_translation(tgt)
    if not t:
        raise RuntimeError(f"argostranslate: no translation object for {source_lang}->en")
    result = t.translate(text)
    if not result:
        raise RuntimeError("argostranslate returned empty string")
    return result



def _backend_mymemory(text: str, source_lang: str) -> str:
    from deep_translator import MyMemoryTranslator  # type: ignore
    code_map = {"hi": "hi-IN", "kn": "kn-IN"}
    src = code_map.get(source_lang, source_lang)
    norm = text.replace("।", ". ").replace("॥", ". ")
    sentences = [s.strip() for s in norm.split(". ") if s.strip()]
    if not sentences:
        sentences = [text]
    parts = []
    for s in sentences:
        chunk = s[:450]
        res = MyMemoryTranslator(source=src, target="en-GB").translate(chunk)
        if res:
            parts.append(res)
    return " ".join(parts)


def _backend_deep_translator(text: str, source_lang: str) -> str:
    from deep_translator import GoogleTranslator  # type: ignore
    CHUNK = 4000

    def _chunk(c: str, tries: int = 3) -> str:
        for i in range(1, tries + 1):
            try:
                r = GoogleTranslator(source=source_lang, target="en").translate(c)
                if r is None:
                    raise RuntimeError("None result")
                return r
            except Exception as e:
                msg = str(e).lower()
                if ("too many" in msg or "429" in msg or "toomany" in msg) and i < tries:
                    w = 3 ** i
                    logger.warning("deep-translator rate limit (attempt %d/%d), waiting %ds...", i, tries, w)
                    time.sleep(w); continue
                raise
        return c

    if len(text) <= CHUNK:
        return _chunk(text)

    norm = text.replace("।", ". ").replace("॥", ". ")
    sents = norm.split(". ")
    chunks, cur = [], ""
    for s in sents:
        cand = (cur + " " if cur else "") + s + ". "
        if len(cand) > CHUNK and cur:
            chunks.append(cur.strip()); cur = s + ". "
        else:
            cur = cand
    if cur.strip():
        chunks.append(cur.strip())

    parts = []
    for i, c in enumerate(chunks):
        parts.append(_chunk(c))
        if i < len(chunks) - 1:
            time.sleep(0.5)
    return " ".join(parts)


def _friendly_error(raw: str) -> str:
    low = raw.lower()
    offline = os.getenv("CLINIQ_TRANSLATION_OFFLINE", "0").lower() in {"1", "true", "yes"}
    if offline:
        return (
            "Offline translation is enabled, but no compatible local translation model "
            "is installed for this language. Set CLINIQ_TRANSLATION_OFFLINE=0 and "
            "restart the API to use the online translation providers."
        )
    if "too many" in low or "429" in low or "rate limit" in low or "toomany" in low:
        return "Google Translate rate limit reached. Please wait 15-20 seconds and try again."
    if ("server error" in low or "connection" in low or "timeout" in low
            or "name resolution" in low or "getaddrinfo" in low):
        return (
            "Translation service is unavailable. Check that this computer has an "
            "internet connection, then try again. Hindi can also work offline when "
            "the Argos Hindi-to-English language package is installed."
        )
    if "identical to input" in low or "untranslated" in low:
        return "Translation did not complete. Please wait and try again."
    return "Translation provider failed. Check internet access and try again."
