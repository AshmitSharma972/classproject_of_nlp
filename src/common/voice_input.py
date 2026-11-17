"""
English-only Speech-to-Text (STT) voice input utility.

Captures microphone audio and transcribes it to English text using
Google Web Speech API via the SpeechRecognition library.

CONSTRAINTS:
- English recognition ONLY (per system requirements).
- Hindi / Kannada voice input is NOT supported.
- This module requires an internet connection for Google STT API.
- SpeechRecognition + pyaudio packages must be installed.
- This module is purely a utility; it contains no application logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceInputResult:
    """Result of a voice transcription attempt."""

    transcribed_text: str
    success: bool
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


LANGUAGE_STT_CODES: dict[str, str] = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Hindi (हिंदी)": "hi-IN",
    "Kannada": "kn-IN",
    "Kannada (ಕನ್ನಡ)": "kn-IN",
}


def transcribe_audio_from_mic(
    language: str = "English",
    timeout_seconds: int = 8,
    phrase_time_limit: int = 30,
) -> VoiceInputResult:
    """
    Record from the default microphone and transcribe speech to text.
    Supports English, Hindi (returns Devanagari script), and Kannada (returns Kannada script).

    Args:
        language: "English", "Hindi", or "Kannada" (with or without script labels).
        timeout_seconds: Max seconds to wait for speech to begin.
        phrase_time_limit: Max seconds to record after speech starts.

    Returns:
        VoiceInputResult with transcribed_text and success flag.
    """
    try:
        import speech_recognition as sr  # type: ignore
    except ImportError:
        msg = (
            "SpeechRecognition is not installed. "
            "Run: pip install SpeechRecognition"
        )
        logger.error(msg)
        return VoiceInputResult(transcribed_text="", success=False, error=msg)

    stt_lang = LANGUAGE_STT_CODES.get(language, "en-IN")

    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise (1s)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            logger.info(
                "Listening for %s speech (lang=%s, timeout=%ds, phrase_limit=%ds)...",
                language,
                stt_lang,
                timeout_seconds,
                phrase_time_limit,
            )
            audio = recognizer.listen(
                source,
                timeout=timeout_seconds,
                phrase_time_limit=phrase_time_limit,
            )

        logger.info("Audio captured, sending to Google STT API (lang=%s)...", stt_lang)
        text = recognizer.recognize_google(audio, language=stt_lang)

        logger.info("Transcription successful | length=%d chars", len(text))
        return VoiceInputResult(transcribed_text=text, success=True)

    except Exception as exc:  # noqa: BLE001
        error_type = type(exc).__name__
        msg = _friendly_error_message(error_type, str(exc))
        logger.warning("Voice input failed: %s | %s", error_type, exc)
        return VoiceInputResult(transcribed_text="", success=False, error=msg)


def transcribe_english_audio(
    timeout_seconds: int = 8,
    phrase_time_limit: int = 30,
) -> VoiceInputResult:
    """Backward-compatible helper for English microphone transcription."""
    return transcribe_audio_from_mic(
        language="English",
        timeout_seconds=timeout_seconds,
        phrase_time_limit=phrase_time_limit,
    )


def _friendly_error_message(error_type: str, raw_msg: str) -> str:
    """Convert SpeechRecognition exceptions to user-friendly messages."""
    messages = {
        "WaitTimeoutError": (
            "No speech detected within the timeout window. "
            "Please try again and speak clearly into the microphone."
        ),
        "UnknownValueError": (
            "Could not understand the audio. "
            "Please speak clearly in English and try again."
        ),
        "RequestError": (
            "Google Speech API unavailable. "
            "Please check your internet connection and try again."
        ),
        "OSError": (
            "Microphone not found or access denied. "
            "Please check that a microphone is connected and accessible."
        ),
    }
    return messages.get(error_type, f"Voice input error: {raw_msg}")


def is_microphone_available() -> bool:
    """
    Check whether a microphone is available on this system.
    Safe to call without raising; returns False on any error.
    """
    try:
        import speech_recognition as sr  # type: ignore
        import pyaudio  # type: ignore  # noqa: F401

        mics = sr.Microphone.list_microphone_names()
        return len(mics) > 0
    except Exception:  # noqa: BLE001
        return False


def list_available_microphones() -> list[str]:
    """Return a list of available microphone device names (empty if none)."""
    try:
        import speech_recognition as sr  # type: ignore
        return sr.Microphone.list_microphone_names()
    except Exception:  # noqa: BLE001
        return []
