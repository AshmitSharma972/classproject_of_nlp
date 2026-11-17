"""LEGACY summary implementation; the active template summary lives in
``src.summarization.summarizer``. This module contains the older Hugging Face
abstractive summarizer and extractive fallback and is not called by the live
ClinicalPipeline path.
"""

from typing import List, Dict, Any, Optional
from config import PRIMARY_SUMMARIZATION_MODEL, FALLBACK_SUMMARIZATION_MODEL
from src.utils import get_logger, SummarizationError, clean_text_whitespace

logger = get_logger("Summarizer")


class MedicalSummarizer:
    """Generates concise, factual clinical summaries using Abstractive or Extractive approaches."""

    def __init__(self, prefer_transformer: bool = True):
        self.prefer_transformer = prefer_transformer
        self.pipeline = None
        self.model_name = None

        if prefer_transformer:
            self._init_transformer_pipeline()

    def _init_transformer_pipeline(self):
        """Initializes Hugging Face transformers summarization pipeline."""
        try:
            from transformers import pipeline
            for model in [PRIMARY_SUMMARIZATION_MODEL, FALLBACK_SUMMARIZATION_MODEL]:
                try:
                    logger.info(f"Attempting to load transformer model: {model}")
                    self.pipeline = pipeline("summarization", model=model)
                    self.model_name = model
                    logger.info(f"Successfully loaded transformer model: {model}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load transformer model '{model}': {e}")
        except ImportError:
            logger.warning("Hugging Face 'transformers' package is not installed. Using Extractive Clinical Summarizer.")

    def generate_summary(
        self,
        full_text: str,
        sentences: List[str],
        key_points: Optional[List[Dict[str, Any]]] = None,
        max_length: int = 150,
        min_length: int = 40
    ) -> Dict[str, Any]:
        """
        Generates a concise summary. Uses abstractive transformer if loaded, otherwise
        falls back to extractive clinical sentence graph summarization.

        Returns:
            Dict with summary_text, mode ('abstractive' or 'extractive'), model_used, and length.
        """
        cleaned_text = clean_text_whitespace(full_text)
        if not cleaned_text:
            return {
                "summary_text": "No text provided for summarization.",
                "mode": "none",
                "model_used": "none",
                "char_length": 0
            }

        # Try Abstractive Transformer pipeline
        if self.pipeline:
            try:
                # Truncate input text to fit standard model token limit (~1024 tokens)
                input_snippet = cleaned_text[:3500]
                summary_output = self.pipeline(
                    input_snippet,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summary_text = summary_output[0]["summary_text"].strip()

                return {
                    "summary_text": summary_text,
                    "mode": "Abstractive (Transformer)",
                    "model_used": self.model_name,
                    "char_length": len(summary_text)
                }
            except Exception as e:
                logger.warning(f"Transformer summarization failed: {e}. Falling back to Extractive Summarizer.")

        # Extractive Clinical Fallback Summarizer
        summary_text = self._extractive_clinical_summary(sentences, key_points)
        return {
            "summary_text": summary_text,
            "mode": "Extractive (Clinical Priority Graph)",
            "model_used": "Extractive Clinical Heuristic",
            "char_length": len(summary_text)
        }

    def _extractive_clinical_summary(
        self,
        sentences: List[str],
        key_points: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Constructs an extractive summary by selecting the highest scoring clinical sentences
        in their original document sequence.
        """
        if not sentences:
            return "Document contains no extractable sentences."

        if key_points:
            # Re-sort top key points into original document order
            ordered_points = sorted(key_points, key=lambda x: x.get("sentence_idx", 0))
            selected_sents = [kp["sentence"] for kp in ordered_points]
            return " ".join(selected_sents)

        # Fallback to first few non-empty sentences
        return " ".join(sentences[:min(4, len(sentences))])
