"""
Main Application CLI: AI-Based Clinical Patient Case-Taking and History Understanding System.

Runs the 9-stage modular clinical pipeline over unstructured patient narration,
displaying:
- Cleaned text
- Entities
- Relations
- Negations/assertions
- Timeline
- Sections
- Patient JSON
- Summary
- Review actions

Provides --legacy mode for previous document-level summarization runner.
"""

from __future__ import annotations

import argparse
import json
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path
from typing import Optional

from src.pipeline import ClinicalPipeline, PipelineResult
from src.review import PhysicianReviewEngine, DISCLAIMER_NOTICE
from src.utils import get_logger

logger = get_logger("MainApp")


def format_cli_output(result: PipelineResult, review_engine: PhysicianReviewEngine) -> str:
    """Format the full pipeline outputs into a clean CLI display string."""
    out = []
    sep = "=" * 80
    subsep = "-" * 80

    out.append("\n" + sep)
    out.append("AI-BASED CLINICAL PATIENT CASE-TAKING AND HISTORY UNDERSTANDING SYSTEM")
    out.append(f"NOTICE: {DISCLAIMER_NOTICE}")
    out.append(sep)

    # 1. Cleaned / Preprocessed Text
    out.append("\n[1. PREPROCESSED & NORMALIZED TEXT]")
    out.append(subsep)
    out.append(f"Raw Input Length       : {len(result.raw_text)} chars")
    out.append(f"Normalized Text        : {result.preprocessing.normalized_text}")
    out.append(f"Segmented Sentences ({len(result.preprocessing.sentences)}):")
    for s in result.preprocessing.sentences:
        out.append(f"  [{s.sentence_id}] (chars {s.raw_start}-{s.raw_end}): {s.text}")
    if result.preprocessing.spell_flags:
        out.append(f"Spelling Warnings Flagged: {result.preprocessing.spell_flags}")

    # 2. Extracted Entities
    out.append("\n[2. EXTRACTED CLINICAL ENTITIES (NER)]")
    out.append(subsep)
    out.append(f"{'Entity Text':<30} | {'Type':<16} | {'Confidence':<10} | {'Raw Span':<12}")
    out.append(subsep)
    for ent in result.entities:
        span_str = f"{ent.span.start}:{ent.span.end}"
        out.append(f"{ent.span.text:<30} | {ent.type.value:<16} | {ent.confidence:<10.2f} | {span_str:<12}")

    # 3. Assertions & Negation
    out.append("\n[3. ASSERTIONS & CONTEXT ANALYSIS]")
    out.append(subsep)
    out.append(f"{'Entity Text':<28} | {'Status':<12} | {'Experiencer':<12} | {'Trigger'}")
    out.append(subsep)
    ent_map = {e.id: e for e in result.entities}
    for ass in result.assertions:
        ent = ent_map.get(ass.entity_id)
        text = ent.span.text if ent else "Unknown"
        trig = f"'{ass.trigger_text}'" if ass.trigger_text else "-"
        out.append(f"{text:<28} | {ass.status.value:<12} | {ass.experiencer.value:<12} | {trig}")

    # 4. Relations
    out.append("\n[4. CLINICAL RELATIONS]")
    out.append(subsep)
    if result.relations:
        out.append(f"{'Head Entity':<25} | {'Relation Type':<22} | {'Tail Entity':<25}")
        out.append(subsep)
        for rel in result.relations:
            head = ent_map.get(rel.head_entity_id)
            tail = ent_map.get(rel.tail_entity_id)
            h_txt = head.span.text if head else rel.head_entity_id
            t_txt = tail.span.text if tail else rel.tail_entity_id
            out.append(f"{h_txt:<25} | {rel.relation_type.value:<22} | {t_txt:<25}")
    else:
        out.append("  (No intra-sentence clinical relations identified)")

    # 5. Timeline
    out.append("\n[5. CLINICAL TIMELINE]")
    out.append(subsep)
    if result.timeline:
        for t in result.timeline:
            out.append(f"  Step {t.order + 1}: {t.label}")
    else:
        out.append("  (No temporal anchors or sequence expressions found)")

    # 6. Section Classification
    out.append("\n[6. SECTION CLASSIFICATION]")
    out.append(subsep)
    for section_name, items in result.patient_record.sections.items():
        out.append(f"  [{section_name}]")
        for it in items:
            out.append(f"    * {it}")

    # 7. Summary
    out.append("\n[7. PHYSICIAN-READY CLINICAL SUMMARY]")
    out.append(result.summary)

    # 8. Physician Review Actions Demo
    out.append("\n[8. PHYSICIAN REVIEW AUDIT]")
    out.append(subsep)
    # Perform sample review actions
    if result.entities:
        first_ent = result.entities[0]
        review_engine.accept(item_id=first_ent.id, reviewer="Dr. Smith, MD", note="Verified accurate extraction")
        review_engine.verify(item_id=first_ent.id, reviewer="Dr. Smith, MD")

    for act in review_engine.get_audit_log():
        out.append(f"  [{act.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {act.action.value} by {act.reviewer} on Item {act.item_id}: Note='{act.note}'")

    # 9. JSON Serialization verification
    out.append("\n[9. STRUCTURED PATIENT RECORD (JSON PREVIEW)]")
    out.append(subsep)
    json_dump = result.patient_record.model_dump_json(indent=2)
    # Preview first 500 chars + summary of json
    out.append(json_dump[:600] + f"\n... [Total JSON length: {len(json_dump)} bytes]")
    out.append(sep + "\n")

    return "\n".join(out)


def run_clinical_case(text: str):
    """Execute the pipeline on input text and print results."""
    pipeline = ClinicalPipeline()
    result = pipeline.run(text)
    review_engine = pipeline.create_review_engine(result.patient_record)
    output_str = format_cli_output(result, review_engine)
    print(output_str)


def run_legacy_flow(args):
    """Execute legacy document-level pipeline for backward compatibility."""
    try:
        from src.document_processor import DocumentProcessor
        from src.text_preprocessor import TextPreprocessor
        from src.information_extractor import InformationExtractor
        from src.key_point_extractor import KeyPointExtractor
        from src.summarizer import MedicalSummarizer
        from src.evidence_linker import EvidenceLinker
        from src.report_generator import ReportGenerator
        from evaluation.evaluator import MedicalNLPEvaluator
        from config import SAMPLE_DIR, PROCESSED_DIR
        print("[Legacy Mode] Running document processor pipeline...")
        # Run evaluation if requested
        if args.evaluate:
            evaluator = MedicalNLPEvaluator()
            print("Running benchmark evaluation over reference summaries...")
        else:
            default_sample = SAMPLE_DIR / "discharge_summary_01.txt"
            if default_sample.exists():
                dp = DocumentProcessor()
                meta = dp.extract_from_file(default_sample)
                print(f"Loaded legacy sample document: {default_sample.name}")
                print(f"Text length: {len(meta['text'])} chars")
    except Exception as e:
        logger.error(f"Legacy flow error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Clinical Patient Case-Taking and History Understanding System"
    )
    parser.add_argument(
        "-t", "--text", type=str, help="Plain English patient narration text to process"
    )
    parser.add_argument(
        "-f", "--file", type=str, help="Path to plain text file containing patient narration"
    )
    parser.add_argument(
        "-c", "--case", type=int, help="Sample case number to run (1 to 20 from data/sample/)"
    )
    parser.add_argument(
        "--legacy", action="store_true", help="Run legacy document processing pipeline"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation benchmark"
    )

    args = parser.parse_args()

    if args.evaluate:
        from evaluation.case_evaluator import CaseEvaluator
        evaluator = CaseEvaluator()
        results = evaluator.evaluate_all()
        evaluator.print_benchmark_report(results)
        return

    if args.legacy:
        run_legacy_flow(args)
        return

    text_to_process: Optional[str] = None

    if args.text:
        text_to_process = args.text
    elif args.file:
        p = Path(args.file)
        if p.exists():
            text_to_process = p.read_text(encoding="utf-8")
        else:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    elif args.case:
        case_file = Path("data") / "sample" / f"case_{args.case:02d}.txt"
        if case_file.exists():
            print(f"Loading sample case from: {case_file}")
            text_to_process = case_file.read_text(encoding="utf-8")
        else:
            print(f"Error: Sample case file '{case_file}' not found.")
            sys.exit(1)
    else:
        # Default sample case if no arguments provided
        default_case = Path("data") / "sample" / "case_01.txt"
        if default_case.exists():
            print(f"No text provided. Running demonstration case ({default_case.name})...")
            text_to_process = default_case.read_text(encoding="utf-8")
        else:
            # Fallback inline test text
            text_to_process = "I've had fever and cough for 5 days. I don't have diabetes or chest pain. My father has hypertension."

    run_clinical_case(text_to_process)


if __name__ == "__main__":
    main()
