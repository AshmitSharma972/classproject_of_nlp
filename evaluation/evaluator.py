"""
Evaluation Engine: Computes quantitative NLP metrics (Precision, Recall, F1 for NER,
and ROUGE-1/ROUGE-2/ROUGE-L for Summarization) alongside qualitative clinical rubrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re

from config import REFERENCE_SUMMARIES_FILE
from src.utils import get_logger

logger = get_logger("Evaluator")


class MedicalNLPEvaluator:
    """Evaluates Medical NER and Summarization modules against ground truth benchmarks."""

    def __init__(self, reference_file: Path = REFERENCE_SUMMARIES_FILE):
        self.reference_data = self._load_reference_data(reference_file)

    def _load_reference_data(self, file_path: Path) -> Dict[str, Any]:
        """Loads reference summaries and entity annotations from JSON benchmark."""
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read reference file {file_path}: {e}")
        return {}

    # --- 1. Information Extraction Evaluation (Precision, Recall, F1) ---

    def evaluate_ner(self, extracted_entities: List[Dict[str, Any]], reference_entities: List[str]) -> Dict[str, float]:
        """
        Calculates Precision, Recall, and F1 score for extracted entities.
        Matches entity strings using lower-case token containment.
        """
        if not reference_entities:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        pred_texts = set((e.get("entity_text") or e.get("text", "")).lower() for e in extracted_entities if e.get("entity_text") or e.get("text"))
        ref_texts = set(t.lower() for t in reference_entities)

        if not pred_texts:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        # True Positives: predicted entities that partially or fully overlap with reference
        true_positives = 0
        for pred in pred_texts:
            if any(pred in ref or ref in pred for ref in ref_texts):
                true_positives += 1

        precision = true_positives / len(pred_texts) if pred_texts else 0.0
        recall = true_positives / len(ref_texts) if ref_texts else 0.0

        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": true_positives,
            "predicted_count": len(pred_texts),
            "reference_count": len(ref_texts)
        }

    # --- 2. Summarization Evaluation (ROUGE Metrics) ---

    def evaluate_summarization(self, generated_summary: str, reference_summary: str) -> Dict[str, float]:
        """
        Computes ROUGE-1, ROUGE-2, and ROUGE-L metrics.
        Uses `rouge_score` package if available, with built-in pure-Python fallback.
        """
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_summary, generated_summary)
            return {
                "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
                "rouge2_f1": round(scores['rouge2'].fmeasure, 4),
                "rougeL_f1": round(scores['rougeL'].fmeasure, 4)
            }
        except ImportError:
            return self._compute_rouge_fallback(generated_summary, reference_summary)

    def _compute_rouge_fallback(self, candidate: str, reference: str) -> Dict[str, float]:
        """Pure Python fallback calculation of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
        cand_tokens = re.findall(r"\w+", candidate.lower())
        ref_tokens = re.findall(r"\w+", reference.lower())

        # ROUGE-1 (Unigrams)
        r1_f1 = self._n_gram_f1(cand_tokens, ref_tokens, n=1)
        # ROUGE-2 (Bigrams)
        r2_f1 = self._n_gram_f1(cand_tokens, ref_tokens, n=2)
        # ROUGE-L (LCS)
        rL_f1 = self._lcs_f1(cand_tokens, ref_tokens)

        return {
            "rouge1_f1": round(r1_f1, 4),
            "rouge2_f1": round(r2_f1, 4),
            "rougeL_f1": round(rL_f1, 4)
        }

    def _n_gram_f1(self, cand_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Calculates n-gram overlap F1 score."""
        def get_ngrams(tokens, k):
            return [tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]

        cand_ngrams = get_ngrams(cand_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        if not cand_ngrams or not ref_ngrams:
            return 0.0

        matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
        p = matches / len(cand_ngrams)
        r = matches / len(ref_ngrams)

        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def _lcs_f1(self, cand_tokens: List[str], ref_tokens: List[str]) -> float:
        """Calculates Longest Common Subsequence (LCS) F1 score."""
        m, n = len(cand_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return 0.0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        p = lcs_len / m
        r = lcs_len / n
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # --- 3. Qualitative Rubric Assessment ---

    def evaluate_qualitative_aspects(self, summary_text: str, source_text: str, extracted_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assesses Qualitative criteria: Relevance, Conciseness, Factual Consistency, and Coverage.
        """
        source_words = len(source_text.split())
        summary_words = len(summary_text.split())

        compression_ratio = round((1.0 - (summary_words / max(source_words, 1))) * 100, 2)
        conciseness_score = "High" if summary_words < 150 else ("Moderate" if summary_words < 250 else "Low")

        # Factual consistency check: ratio of summary words present in source
        sum_words_set = set(re.findall(r"\w+", summary_text.lower()))
        src_words_set = set(re.findall(r"\w+", source_text.lower()))

        grounded_words = sum_words_set.intersection(src_words_set)
        factual_grounding_rate = round(len(grounded_words) / max(len(sum_words_set), 1), 4)

        return {
            "source_word_count": source_words,
            "summary_word_count": summary_words,
            "compression_ratio_pct": compression_ratio,
            "conciseness_rating": conciseness_score,
            "factual_grounding_rate": factual_grounding_rate,
            "factual_consistency": "High" if factual_grounding_rate > 0.85 else "Moderate",
            "coverage": "High" if summary_words >= 30 else "Basic"
        }
