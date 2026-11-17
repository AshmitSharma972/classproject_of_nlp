"""
Clinical Pipeline Orchestrator.

Executes exactly the 9 stages in order:
1. Preprocessing
2. NER
3. Assertion/Negation
4. Temporal
5. Relation Extraction
6. Section Classification
7. Patient Representation
8. Summary
9. Physician Review

Invariants:
- Modules do not contain orchestration logic.
- Pipeline orchestrator connects modules by passing structured objects.
- Logs intermediate results after every stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.common.schemas import (
    Entity,
    Assertion,
    TimeExpr,
    Relation,
    SectionItem,
    TimelineEntry,
    PatientRecord,
    PreprocessingOutput,
)
from src.preprocessing import ClinicalPreprocessor
from src.ner import ClinicalNER
from src.negation import AssertionEngine
from src.temporal import TemporalEngine
from src.relation_extraction import RelationEngine
from src.section_classification import SectionClassifier
from src.patient_representation import PatientRecordBuilder
from src.summarization import ClinicalSummarizer
from src.review import PhysicianReviewEngine
from src.utils import get_logger

logger = get_logger("ClinicalPipeline")


class PipelineResult(BaseModel):
    """Encapsulates the output of all stages of the clinical pipeline."""
    raw_text: str
    preprocessing: PreprocessingOutput
    entities: List[Entity]
    assertions: List[Assertion]
    time_expressions: List[TimeExpr]
    relations: List[Relation]
    section_items: List[SectionItem]
    timeline: List[TimelineEntry]
    patient_record: PatientRecord
    summary: str

    model_config = {"arbitrary_types_allowed": True}


class ClinicalPipeline:
    """
    Modular 9-stage clinical NLP pipeline for patient case-taking.
    """

    def __init__(self):
        logger.info("Initializing ClinicalPipeline stages...")
        self.preprocessor = ClinicalPreprocessor()
        self.ner_engine = ClinicalNER()
        self.assertion_engine = AssertionEngine()
        self.temporal_engine = TemporalEngine()
        self.relation_engine = RelationEngine()
        self.section_classifier = SectionClassifier()
        self.patient_builder = PatientRecordBuilder()
        self.summarizer = ClinicalSummarizer()
        logger.info("All 9 clinical pipeline stages initialized successfully.")

    def run(self, raw_text: str) -> PipelineResult:
        """
        Executes the 9 pipeline stages sequentially over raw patient text.

        Args:
            raw_text: Unstructured raw clinical/patient narration.

        Returns:
            PipelineResult containing intermediate and final structured artifacts.
        """
        logger.info("=== Starting Clinical Pipeline Execution ===")
        logger.info("Stage 1/9: Preprocessing")
        prep_output = self.preprocessor.run(raw_text)
        logger.info("Stage 1 Output: %d sentences, %d tokens",
                    len(prep_output.sentences),
                    prep_output.metadata.get("token_count", 0))

        logger.info("Stage 2/9: Named Entity Recognition")
        entities = self.ner_engine.run(prep_output)
        logger.info("Stage 2 Output: %d clinical entities extracted", len(entities))

        logger.info("Stage 3/9: Assertion and Negation Analysis")
        assertions = self.assertion_engine.run(entities, prep_output)
        logger.info("Stage 3 Output: %d assertions analyzed", len(assertions))

        logger.info("Stage 4/9: Temporal Processing and Timeline Building")
        time_expressions, timeline = self.temporal_engine.run(entities, assertions, prep_output)
        logger.info("Stage 4 Output: %d time expressions, %d timeline nodes",
                    len(time_expressions), len(timeline))

        logger.info("Stage 5/9: Relation Extraction")
        relations = self.relation_engine.run(entities, assertions, time_expressions, prep_output)
        logger.info("Stage 5 Output: %d clinical relations extracted", len(relations))

        logger.info("Stage 6/9: Section Classification")
        section_items = self.section_classifier.run(entities, assertions, time_expressions, relations)
        logger.info("Stage 6 Output: %d section items assigned", len(section_items))

        logger.info("Stage 7/9: Patient Representation Assembly")
        patient_record = self.patient_builder.build(
            raw_text=raw_text,
            preprocessing_output=prep_output,
            entities=entities,
            assertions=assertions,
            time_expressions=time_expressions,
            relations=relations,
            section_items=section_items,
            timeline=timeline,
        )
        logger.info("Stage 7 Output: PatientRecord %s assembled", patient_record.id)

        logger.info("Stage 8/9: Clinical Summarization")
        summary = self.summarizer.summarize(patient_record)
        logger.info("Stage 8 Output: Summary generated (%d characters)", len(summary))

        logger.info("Stage 9/9: Physician Review Readiness")
        # Review engine is instantiated attached to the generated patient_record
        _ = PhysicianReviewEngine(patient_record)
        logger.info("Stage 9 Output: PhysicianReviewEngine attached and ready")

        logger.info("=== Clinical Pipeline Execution Complete ===")

        return PipelineResult(
            raw_text=raw_text,
            preprocessing=prep_output,
            entities=entities,
            assertions=assertions,
            time_expressions=time_expressions,
            relations=relations,
            section_items=section_items,
            timeline=timeline,
            patient_record=patient_record,
            summary=summary,
        )

    def create_review_engine(self, patient_record: PatientRecord) -> PhysicianReviewEngine:
        """Create a PhysicianReviewEngine instance for human-in-the-loop validation."""
        return PhysicianReviewEngine(patient_record)
