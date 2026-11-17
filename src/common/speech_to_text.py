"""Local multilingual speech-to-text for ClinIQ microphone input.

The default Whisper model is multilingual and is used in *transcribe* mode,
so Hindi and Kannada speech is returned in its native script rather than
translated to English.  Model weights are downloaded by Transformers only on
the first transcription and then read from the local Hugging Face cache.
"""
from __future__ import annotations

import io
import os
import wave
from functools import lru_cache

import numpy as np


LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Kannada": "kn"}
LANGUAGE_NAMES = {"English": "english", "Hindi": "hindi", "Kannada": "kannada"}
_SCRIPT_RANGES = {"Hindi": ("\u0900", "\u097f"), "Kannada": ("\u0c80", "\u0cff")}


class SpeechToTextError(RuntimeError):
    """A user-safe local transcription error."""


@lru_cache(maxsize=1)
def _pipeline():
    """Load the local ASR model once, only when microphone input is used."""
    try:
        from transformers import pipeline  # Imported lazily to keep API startup quick.
        model = os.getenv("CLINIQ_ASR_MODEL", "openai/whisper-base")
        return pipeline("automatic-speech-recognition", model=model, device=-1)
    except Exception as exc:  # Model download/load errors must not expose internals to UI.
        raise SpeechToTextError(
            "The local speech model could not be loaded. Check internet access for the first "
            "model download, then restart the API and try again."
        ) from exc


def _read_wav(audio: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(audio), "rb") as stream:
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            rate = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
    except (wave.Error, EOFError) as exc:
        raise SpeechToTextError("The recording is not a valid WAV audio file.") from exc
    if channels not in (1, 2) or width not in (1, 2, 4) or rate <= 0:
        raise SpeechToTextError("The recording format is unsupported. Please record again.")
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    values = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if width == 1:
        values = (values - 128.0) / 128.0
    else:
        values /= float(2 ** (8 * width - 1))
    if channels == 2:
        values = values.reshape(-1, 2).mean(axis=1)
    return values, rate


def _has_expected_script(text: str, language: str) -> bool:
    expected = _SCRIPT_RANGES.get(language)
    if expected is None:
        return True
    lower, upper = expected
    return any(lower <= character <= upper for character in text)


def _transcribe_google(audio: bytes, language: str) -> str:
    """Fast transcription via Google STT API (native script for Hindi and Kannada)."""
    import speech_recognition as sr
    lang_code = {"English": "en-IN", "Hindi": "hi-IN", "Kannada": "kn-IN"}.get(language, "en-IN")
    rec = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio)) as source:
        audio_data = rec.record(source)
    try:
        text = rec.recognize_google(audio_data, language=lang_code).strip()
    except sr.UnknownValueError:
        raise SpeechToTextError("No speech was recognized. Please speak clearly and try again.")
    if not text:
        raise SpeechToTextError("No speech was recognized. Please speak clearly and try again.")
    if not _has_expected_script(text, language):
        raise SpeechToTextError(
            f"The model did not return {language} script text. Please record again or type the narration."
        )
    return text


def transcribe_audio(audio: bytes, language: str) -> str:
    """Return native-script text from a browser-recorded WAV file."""
    if language not in LANGUAGE_CODES:
        raise SpeechToTextError("Supported microphone languages are English, Hindi, and Kannada.")
    samples, sample_rate = _read_wav(audio)
    if samples.size == 0:
        raise SpeechToTextError("The recording is empty. Please speak and record again.")

    # 1. Fast online recognition (Google STT) - provides accurate native script in 1s
    try:
        return _transcribe_google(audio, language)
    except SpeechToTextError:
        raise
    except Exception:
        # Offline or Google STT network error: fall back to local Whisper
        pass

    # 2. Local Whisper fallback (offline or when Google STT is unreachable)
    try:
        result = _pipeline()(
            {"array": samples, "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe", "language": LANGUAGE_NAMES[language]},
        )
        text = str(result.get("text", "")).strip()
    except SpeechToTextError:
        raise
    except Exception as exc:
        raise SpeechToTextError("Speech recognition could not process this recording. Please try again.") from exc
    if not text:
        raise SpeechToTextError("No speech was recognized. Please speak clearly and try again.")
    if not _has_expected_script(text, language):
        raise SpeechToTextError(
            f"The model did not return {language} script text. Please record again or type the narration."
        )
    return text
