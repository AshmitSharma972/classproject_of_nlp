"""
Text Preprocessor Module: Provides sentence segmentation, tokenization, normalization,
and medical-safe stop word filtering while retaining negation and temporal context.
"""

import re
import unicodedata
from typing import List, Dict, Any
from src.utils import get_logger, clean_text_whitespace

logger = get_logger("TextPreprocessor")

# Medical Critical Words that MUST NEVER be removed as stopwords
CRITICAL_MEDICAL_MODIFIERS = {
    "no", "not", "none", "neither", "nor", "without", "denies", "denied",
    "absent", "free", "clear", "never", "rules", "ruled", "out", "negative",
    "history", "past", "prior", "before", "after", "currently", "now", "presently",
    "severe", "mild", "moderate", "acute", "chronic", "increased", "decreased",
    "elevated", "high", "low", "normal", "abnormal", "positive"
}


class TextPreprocessor:
    """Handles clinical text cleaning, sentence segmentation, and medical-safe tokenization."""

    def __init__(self):
        self._init_nltk_and_spacy()

    def _init_nltk_and_spacy(self):
        """Initializes optional NLTK or spaCy segmenters."""
        self.nltk_sent = None
        self.spacy_nlp = None

        try:
            import nltk
            try:
                self.nltk_sent = nltk.sent_tokenize
            except LookupError:
                nltk.download("punkt", quiet=True)
                self.nltk_sent = nltk.sent_tokenize
        except Exception:
            pass

        try:
            import spacy
            try:
                self.spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
            except Exception:
                pass
        except Exception:
            pass

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segments raw clinical text into clean individual sentences.
        Uses NLTK/spaCy if available, with regex sentence boundary fallback.
        """
        cleaned = clean_text_whitespace(text)
        if not cleaned:
            return []

        # 1. Try NLTK
        if self.nltk_sent:
            try:
                raw_sentences = self.nltk_sent(cleaned)
                sentences = [s.strip() for s in raw_sentences if s.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                logger.debug(f"NLTK sentence segmentation fallback: {e}")

        # 2. Try spaCy
        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(cleaned[:100000])
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as e:
                logger.debug(f"spaCy sentence segmentation fallback: {e}")

        # 3. Robust Regex Fallback
        return self._regex_sentence_split(cleaned)

    def _regex_sentence_split(self, text: str) -> List[str]:
        """Splits text on sentence boundaries considering clinical abbreviations (e.g., Dr., mg., v.s., c/o)."""
        # Protect common medical abbreviations
        protected = text
        protected = re.sub(r"\b(Dr|Cr|Pt|Mr|Mrs|Ms|Vs|No|Tab|Cap|Inj|Vol|approx|w/o|s/p|c/o)\.", r"\1_DOT_", protected)
        protected = re.sub(r"\b(\d+)\.(\d+)", r"\1_DECIMAL_\2", protected)  # e.g. 3.5 mg

        # Split on sentence ending punctuation
        raw_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\n])|\n\n+", protected)

        sentences = []
        for s in raw_splits:
            s_restored = s.replace("_DOT_", ".").replace("_DECIMAL_", ".").strip()
            if s_restored:
                sentences.append(s_restored)

        return sentences

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenizes text into word tokens using medical-aware regex."""
        # Retain hyphenated medical terms (e.g. anti-inflammatory, non-smoker) and slash terms (c/o, s/p)
        tokens = re.findall(r"\b[\w/'-]+\b", text)
        return tokens

    def normalize_text(self, text: str) -> str:
        """Applies unicode normalization and cleans illegal characters."""
        if not text:
            return ""
        # Unicode NFKD normalization
        normalized = unicodedata.normalize("NFKD", text)
        # Remove control characters except newlines/tabs
        cleaned = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r")
        return clean_text_whitespace(cleaned)

    def filter_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Filters non-essential stop words while strictly preserving critical medical
        negations, temporal markers, and severity modifiers.
        """
        standard_stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
            "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
            "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
            "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
            "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
            "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
            "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
            "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
            "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
            "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
            "you've", "your", "yours", "yourself", "yourselves"
        }

        # Keep words that are in CRITICAL_MEDICAL_MODIFIERS
        medical_safe_stopwords = standard_stopwords - CRITICAL_MEDICAL_MODIFIERS

        filtered = [t for t in tokens if t.lower() not in medical_safe_stopwords]
        return filtered

    def preprocess_pipeline(self, raw_text: str) -> Dict[str, Any]:
        """
        Runs the full text preprocessing pipeline.

        Returns:
            Dict containing normalized_text, sentences, tokens, and medical_tokens.
        """
        normalized = self.normalize_text(raw_text)
        sentences = self.segment_sentences(normalized)
        tokens = self.tokenize_words(normalized)
        medical_tokens = self.filter_stop_words(tokens)

        return {
            "normalized_text": normalized,
            "sentences": sentences,
            "sentence_count": len(sentences),
            "tokens": tokens,
            "token_count": len(tokens),
            "medical_tokens": medical_tokens
        }
