"""
Streamlit Web Application: AI-Based Clinical Patient Case-Taking and History Understanding System.

Provides an interactive web interface for:
1. 9-Stage Modular Clinical Case-Taking Pipeline
2. Multilingual Input Support (English, Hindi, Kannada) with auto-translation to English
3. English-only Speech-to-Text (voice input) via microphone
4. Live Stage-by-Stage Visualizations (Preproc, NER, Assertion, Temporal, Relations, Sections)
5. Physician-Ready Clinical Summary & Structured PatientRecord JSON
6. Human-in-the-loop Physician Review Engine with Real-time Audit Trail
7. Benchmark Evaluation Suite (Precision, Recall, F1 for NER, Assertion, Relations)
8. Backward-compatible Legacy Document Processing mode
9. Pipeline Architecture Knowledge Graph (graphify-style visualization)
"""

from __future__ import annotations

import json
import os
import sys
import threading

# Reconfigure console streams on Windows to prevent UnicodeEncodeError crashes
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.pipeline import ClinicalPipeline, PipelineResult
from src.common.schemas import (
    AssertionStatus,
    EntityType,
    Experiencer,
    PatientRecord,
    RelationType,
    ReviewActionType,
    SectionType,
)
from src.common.translator import translate_to_english, SUPPORTED_LANGUAGES
from src.common.voice_input import (
    transcribe_english_audio,
    is_microphone_available,
    VoiceInputResult,
)
from src.review import PhysicianReviewEngine, DISCLAIMER_NOTICE
from evaluation.case_evaluator import CaseEvaluator

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Case-Taking & History Understanding",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 18px;
        font-size: 0.92rem;
        font-weight: 600;
        color: #991B1B;
    }
    .card-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .timeline-node {
        border-left: 3px solid #3B82F6;
        padding-left: 12px;
        margin-bottom: 10px;
    }
    .lang-banner {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 12px;
        font-size: 0.90rem;
        color: #1E40AF;
    }
    .translated-box {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.88rem;
        color: #166534;
    }
    .voice-note {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 6px;
    }
    .pipeline-stage {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 8px 14px;
        border-radius: 6px;
        margin: 4px 0;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .arch-arrow {
        color: #6B7280;
        font-size: 1.2rem;
        text-align: center;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_clinical_pipeline() -> ClinicalPipeline:
    """Initialize and cache the 9-stage clinical pipeline."""
    return ClinicalPipeline()


def load_sample_case(case_num: int) -> str:
    """Load sample patient narration from data/sample/case_XX.txt."""
    file_path = PROJECT_ROOT / "data" / "sample" / f"case_{case_num:02d}.txt"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "review_audit_log" not in st.session_state:
    st.session_state.review_audit_log = []
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "selected_case_text" not in st.session_state:
    st.session_state.selected_case_text = (
        "I've had fever... for 5 days!!! I DON'T have diabetes. "
        "My father has hypertension. Currently taking Paracetamol 500 mg twice a day."
    )
if "voice_transcribed_text" not in st.session_state:
    st.session_state.voice_transcribed_text = ""
if "translation_result" not in st.session_state:
    st.session_state.translation_result = None
if "input_language" not in st.session_state:
    st.session_state.input_language = "English"


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    # Header & Mandatory Disclaimer Notice
    st.markdown(
        '<div class="main-header">🩺 AI Clinical Patient Case-Taking & History Understanding</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Modular, explainable clinical NLP pipeline converting plain narration '
        'into structured records and physician-ready summaries. '
        'Supports English, Hindi, and Kannada input with English-only voice capture.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="disclaimer-banner">⚠️ {DISCLAIMER_NOTICE}</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Sidebar Navigation & Controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Navigation & Settings")
        mode = st.radio(
            "Select Mode:",
            [
                "🩺 Case-Taking Pipeline",
                "📊 Benchmark Evaluation",
                "🗺️ Pipeline Architecture",
                "📁 Legacy Document Analyzer",
            ],
            index=0,
        )

        st.divider()

        if mode == "🩺 Case-Taking Pipeline":
            st.subheader("Presets & Sample Cases")
            selected_preset = st.selectbox(
                "Load Synthetic Clinical Case (1-20):",
                [f"Case {i:02d}" for i in range(1, 21)],
                index=0,
            )
            case_num = int(selected_preset.split()[1])

            if st.button("📥 Load Selected Case", width='stretch'):
                loaded_text = load_sample_case(case_num)
                if loaded_text:
                    st.session_state.selected_case_text = loaded_text
                    st.session_state.pipeline_result = None
                    st.session_state.translation_result = None
                    st.session_state.input_language = "English"
                    st.success(f"Loaded Case {case_num:02d} successfully!")

            if st.button("✨ Load Example Prompt Case", width='stretch'):
                st.session_state.selected_case_text = (
                    "I've had fever... for 5 days!!! I DON'T have diabetes. "
                    "My father has hypertension. Currently taking Paracetamol 500 mg twice a day."
                )
                st.session_state.pipeline_result = None
                st.session_state.translation_result = None
                st.session_state.input_language = "English"

            st.divider()
            st.subheader("Input Language")
            lang_choice = st.radio(
                "Select patient narration language:",
                list(SUPPORTED_LANGUAGES.keys()),
                index=list(SUPPORTED_LANGUAGES.keys()).index(
                    st.session_state.input_language
                ),
                help=(
                    "English: full voice + text support\n"
                    "Hindi / Kannada: text input only -- "
                    "auto-translated to English before pipeline."
                ),
            )
            if lang_choice != st.session_state.input_language:
                st.session_state.input_language = lang_choice
                st.session_state.translation_result = None
                st.session_state.pipeline_result = None
                # Clear text area so user types in the new language
                # (don't keep English default text when switching to Hindi/Kannada)
                if lang_choice != "English":
                    st.session_state.selected_case_text = ""
                else:
                    st.session_state.selected_case_text = (
                        "I've had fever... for 5 days!!! I DON'T have diabetes. "
                        "My father has hypertension. Currently taking Paracetamol 500 mg twice a day."
                    )

    # ------------------------------------------------------------------
    # Route according to mode
    # ------------------------------------------------------------------
    if mode == "🩺 Case-Taking Pipeline":
        render_clinical_case_pipeline()
    elif mode == "📊 Benchmark Evaluation":
        render_benchmark_evaluation()
    elif mode == "🗺️ Pipeline Architecture":
        render_pipeline_architecture()
    else:
        render_legacy_analyzer()


# ---------------------------------------------------------------------------
# Case-Taking Pipeline
# ---------------------------------------------------------------------------
def render_clinical_case_pipeline():
    """Render the primary 9-stage clinical case-taking pipeline UI."""
    pipeline = get_clinical_pipeline()
    current_lang = st.session_state.input_language

    st.subheader("1. Patient Narration Input")

    # ------------------------------------------------------------------
    # Language banner
    # ------------------------------------------------------------------
    if current_lang != "English":
        st.markdown(
            f'<div class="lang-banner">'
            f'🌐 <b>Input language: {current_lang}</b> &nbsp;|&nbsp; '
            f'Text will be auto-translated to English before pipeline processing. '
            f'Voice input is available in <b>English mode only</b>.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="lang-banner">'
            '🇬🇧 <b>Input language: English</b> &nbsp;|&nbsp; '
            'Voice (microphone) input enabled for English only.'
            '</div>',
            unsafe_allow_html=True,
        )

    col_input, col_meta = st.columns([3, 1])

    # ------------------------------------------------------------------
    # Determine placeholder / label based on language
    # ------------------------------------------------------------------
    lang_prompts = {
        "English": (
            "Type or paste unstructured patient narrative here...",
            "Enter raw patient narration (English):",
        ),
        "Hindi (हिंदी)": (
            "यहाँ मरीज का विवरण हिंदी में टाइप करें...",
            "रोगी का विवरण हिंदी में दर्ज करें:",
        ),
        "Kannada (ಕನ್ನಡ)": (
            "ಇಲ್ಲಿ ರೋಗಿಯ ವಿವರಣೆಯನ್ನು ಕನ್ನಡದಲ್ಲಿ ನಮೂದಿಸಿ...",
            "ರೋಗಿಯ ವಿವರಣೆಯನ್ನು ಕನ್ನಡದಲ್ಲಿ ನಮೂದಿಸಿ:",
        ),
    }
    placeholder_text, input_label = lang_prompts.get(
        current_lang, lang_prompts["English"]
    )

    with col_input:
        patient_text = st.text_area(
            input_label,
            value=st.session_state.selected_case_text,
            height=160,
            placeholder=placeholder_text,
            key=f"patient_text_{current_lang}",
        )

        # ------------------------------------------------------------------
        # Voice Input (English ONLY)
        # ------------------------------------------------------------------
        if current_lang == "English":
            _render_voice_input_section(patient_text)
            # If voice was just captured, update patient_text
            if st.session_state.voice_transcribed_text:
                patient_text = st.session_state.voice_transcribed_text

    with col_meta:
        st.info(
            "**Pipeline Constraints**\n\n"
            "- Plain English input only (v1 pipeline)\n"
            "- Hindi/Kannada: text input → auto-translation\n"
            "- Voice input: English only\n"
            "- 9-stage sequential deterministic execution\n"
            "- Decision support, never autonomous diagnosis"
        )
        run_btn = st.button(
            "🚀 Run Clinical Pipeline", type="primary", width='stretch'
        )

    # ------------------------------------------------------------------
    # Translation + Pipeline -- single atomic flow
    # ------------------------------------------------------------------
    # For non-English: translation and pipeline run in ONE button-click
    # cycle.  No st.success/st.rerun between them -- that would reset
    # run_btn to False before the pipeline fires.
    # ------------------------------------------------------------------

    translation_msg = ""        # success/error message shown AFTER pipeline
    translation_failed = False

    if current_lang == "English":
        # English auto-runs on first load or explicit button click
        english_text_for_pipeline = patient_text
        # Only auto-run if text is non-empty and contains no non-Latin chars
        # (guards against auto-running with leftover Hindi/Kannada text)
        _has_non_latin = any(
            ("\u0900" <= c <= "\u097F") or ("\u0C80" <= c <= "\u0CFF")
            for c in patient_text
        )
        should_run = (run_btn or (st.session_state.pipeline_result is None)) and bool(patient_text) and not _has_non_latin

    else:
        english_text_for_pipeline = ""
        should_run = False

        if run_btn:
            # ---- 1. Translate ----
            with st.spinner(f"Translating {current_lang} → English…"):
                trans = translate_to_english(patient_text, current_lang)
                st.session_state.translation_result = trans

            if trans.error:
                translation_failed = True
                translation_msg = trans.error
            else:
                # Validate no source-script chars leaked through
                _dev = any('\u0900' <= c <= '\u097F' for c in trans.translated_text)
                _kn  = any('\u0C80' <= c <= '\u0CFF' for c in trans.translated_text)
                if _dev or _kn:
                    translation_failed = True
                    translation_msg = (
                        "Translation returned non-English text. "
                        "Please wait 15 seconds and try again."
                    )
                    st.session_state.translation_result = None
                else:
                    english_text_for_pipeline = trans.translated_text
                    should_run = True   # ← translation OK, pipeline will fire below

        elif (st.session_state.translation_result
              and not st.session_state.translation_result.error):
            # Previous successful translation exists -- keep it visible
            english_text_for_pipeline = st.session_state.translation_result.translated_text

    # ---- 2. Run pipeline (immediately after translation, same render cycle) ----
    if should_run and english_text_for_pipeline:
        step = "Step 2/2 -- " if (run_btn and current_lang != "English") else ""
        with st.spinner(f"{step}Running 9-stage clinical pipeline…"):
            result = pipeline.run(english_text_for_pipeline)
            st.session_state.pipeline_result = result
            st.session_state.selected_case_text = patient_text

    # ---- 3. Show translation status messages AFTER pipeline completes ----
    if translation_failed:
        st.error(
            f"**Translation incomplete:** {translation_msg}\n\n"
            "Please wait a moment and click **🚀 Run Clinical Pipeline** again."
        )
    elif run_btn and current_lang != "English" and not translation_failed:
        if st.session_state.translation_result and st.session_state.translation_result.was_translated:
            t = st.session_state.translation_result
            st.success(
                f"✅ Translated {t.source_language} → English "
                f"({len(t.original_text)} chars → {len(t.translated_text)} chars). "
                f"Pipeline complete."
            )

    # ---- 4. Show translation preview panel ----
    if st.session_state.translation_result and current_lang != "English":
        trans = st.session_state.translation_result
        if trans.was_translated:
            with st.expander("🌐 Translation Preview", expanded=True):
                col_orig, col_eng = st.columns(2)
                with col_orig:
                    st.markdown(f"**Original ({trans.source_language}):**")
                    st.markdown(
                        f'<div class="card-box" style="white-space:pre-wrap;">'
                        f'{trans.original_text}</div>',
                        unsafe_allow_html=True,
                    )
                with col_eng:
                    st.markdown("**Translated (English) -- pipeline input:**")
                    st.markdown(
                        f'<div class="translated-box" style="white-space:pre-wrap;">'
                        f'{trans.translated_text}</div>',
                        unsafe_allow_html=True,
                    )
                st.caption(
                    f"Translated {trans.source_language_code} → en via Google Translate. "
                    "Original preserved for provenance. Report is in English."
                )

    result: PipelineResult = st.session_state.pipeline_result

    # For non-English languages: show a clear prompt when no result yet
    if result is None and current_lang != "English":
        st.info(
            f"**Ready for {current_lang} input.**\n\n"
            f"1. Type your patient narration in {current_lang} in the text box above.\n"
            f"2. Click **Run Clinical Pipeline**.\n\n"
            f"The system will automatically:\n"
            f"- Translate your text from {current_lang} to English\n"
            f"- Run all 9 pipeline stages on the English translation\n"
            f"- Show you the clinical report and summary in English\n\n"
            f"The original {current_lang} text is preserved alongside the translation."
        )

    if result:
        st.divider()

        # Display Stage Tabs
        (
            tab_summary,
            tab_entities,
            tab_assertion,
            tab_temporal,
            tab_relations,
            tab_sections,
            tab_preproc,
            tab_json,
            tab_review,
        ) = st.tabs([
            "📋 Clinical Summary",
            "🏷️ Entities (NER)",
            "⚖️ Assertions & Negation",
            "⏱️ Timeline & Temporal",
            "🔗 Relations",
            "📑 Sections",
            "🧹 Preprocessing",
            "📦 Patient JSON",
            "🩺 Physician Review",
        ])

        # --- TAB 1: Clinical Summary ---
        with tab_summary:
            st.markdown("### Physician-Ready Clinical Summary")
            st.info(
                "Generated exclusively from structured `PatientRecord` data "
                "with strict assertion preservation."
            )

            if current_lang != "English" and st.session_state.translation_result:
                trans = st.session_state.translation_result
                st.markdown(
                    f'<div class="lang-banner">'
                    f'🌐 This report was generated from a <b>{trans.source_language}</b> narration. '
                    f'The pipeline ran on the auto-translated English text. '
                    f'Original {trans.source_language} input is preserved below for reference.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"View original {trans.source_language} text"):
                    st.markdown(
                        f'<div class="card-box" style="white-space:pre-wrap;">'
                        f'{trans.original_text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with st.expander("View English translation (pipeline input)"):
                    st.markdown(
                        f'<div class="translated-box" style="white-space:pre-wrap;">'
                        f'{trans.translated_text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Display summary in a code block (preserves formatting, no encoding issues)
            st.code(result.summary, language=None)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 Download Clinical Summary (.txt)",
                    data=result.summary,
                    file_name=f"clinical_summary_{result.patient_record.id[:8]}.txt",
                    mime="text/plain",
                    width='stretch',
                )
            with col_dl2:
                st.download_button(
                    "📥 Download Structured Patient Record (.json)",
                    data=result.patient_record.model_dump_json(indent=2),
                    file_name=f"patient_record_{result.patient_record.id[:8]}.json",
                    mime="application/json",
                    width='stretch',
                )

        # --- TAB 2: NER Entities ---
        with tab_entities:
            st.markdown("### Stage 2: Named Entity Recognition (NER)")
            st.write(
                f"Total entities extracted: **{len(result.entities)}** "
                f"across 14 supported clinical types."
            )

            if result.entities:
                ent_data = []
                for e in result.entities:
                    ent_data.append({
                        "Entity Text": e.span.text,
                        "Category / Type": e.type.value,
                        "Confidence": f"{e.confidence:.2f}",
                        "Raw Offset Span": f"{e.span.start}:{e.span.end}",
                        "Sentence Index": e.span.source_sentence_id,
                    })
                df_ent = pd.DataFrame(ent_data)

                all_types = ["ALL"] + sorted(list(set(e.type.value for e in result.entities)))
                type_filter = st.selectbox("Filter by Entity Category:", all_types)
                if type_filter != "ALL":
                    df_ent = df_ent[df_ent["Category / Type"] == type_filter]

                st.dataframe(df_ent, width='stretch', hide_index=True)
            else:
                st.warning("No clinical entities detected in the narration.")

        # --- TAB 3: Assertions & Negation ---
        with tab_assertion:
            st.markdown("### Stage 3: Assertion & Negation Context Analysis")
            st.write(
                "Contextual scope analysis distinguishing PRESENT, NEGATED, "
                "POSSIBLE, HISTORICAL, and FAMILY status."
            )

            ent_map = {e.id: e for e in result.entities}
            ass_data = []
            for a in result.assertions:
                ent = ent_map.get(a.entity_id)
                ass_data.append({
                    "Entity": ent.span.text if ent else a.entity_id,
                    "Assertion Status": a.status.value,
                    "Experiencer": a.experiencer.value,
                    "Trigger Phrase": f"'{a.trigger_text}'" if a.trigger_text else "-",
                    "Confidence": f"{a.confidence:.2f}",
                })

            if ass_data:
                df_ass = pd.DataFrame(ass_data)
                status_filter = st.selectbox(
                    "Filter by Status:",
                    ["ALL"] + [s.value for s in AssertionStatus],
                )
                if status_filter != "ALL":
                    df_ass = df_ass[df_ass["Assertion Status"] == status_filter]
                st.dataframe(df_ass, width='stretch', hide_index=True)
            else:
                st.info("No assertions to display.")

        # --- TAB 4: Timeline & Temporal ---
        with tab_temporal:
            st.markdown("### Stage 4: Temporal Expressions & Clinical Timeline")

            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                st.markdown("#### Extracted Temporal Expressions")
                if result.time_expressions:
                    te_data = []
                    for t in result.time_expressions:
                        linked_texts = [
                            ent_map[eid].span.text
                            for eid in t.linked_entity_ids
                            if eid in ent_map
                        ]
                        te_data.append({
                            "Text": t.raw_value,
                            "Type": t.type.value,
                            "Normalized Value": t.normalized_value or "-",
                            "Linked Entities": ", ".join(linked_texts) if linked_texts else "(None)",
                            "Raw Span": f"{t.span.start}:{t.span.end}",
                        })
                    st.dataframe(pd.DataFrame(te_data), width='stretch', hide_index=True)
                else:
                    st.info("No temporal expressions identified.")

            with col_t2:
                st.markdown("#### Chronological Timeline Nodes")
                if result.timeline:
                    for node in result.timeline:
                        st.markdown(
                            f'<div class="timeline-node"><b>Step {node.order + 1}:</b> {node.label}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No chronological anchors available to form a timeline.")

        # --- TAB 5: Relations ---
        with tab_relations:
            st.markdown("### Stage 5: Intra-Sentence Clinical Relations")
            st.write(
                "Extracts relationships between medication-dosage, "
                "medication-frequency, symptom-duration, etc."
            )

            if result.relations:
                rel_data = []
                for r in result.relations:
                    h = ent_map.get(r.head_entity_id)
                    t = ent_map.get(r.tail_entity_id)
                    rel_data.append({
                        "Head Entity": h.span.text if h else r.head_entity_id,
                        "Relation Type": r.relation_type.value,
                        "Tail Entity": t.span.text if t else r.tail_entity_id,
                        "Sentence ID": r.source_sentence_id,
                        "Evidence Sentence": r.evidence_text or "-",
                    })
                st.dataframe(pd.DataFrame(rel_data), width='stretch', hide_index=True)
            else:
                st.info("No intra-sentence clinical relations found.")

        # --- TAB 6: Sections ---
        with tab_sections:
            st.markdown("### Stage 6: Section Classification (8 Clinical Note Sections)")
            sections = result.patient_record.sections
            if sections:
                for sec_name, items in sections.items():
                    with st.expander(f"📌 {sec_name} ({len(items)} items)", expanded=True):
                        for it in items:
                            st.markdown(f"- {it}")
            else:
                st.info("No section items classified.")

        # --- TAB 7: Preprocessing ---
        with tab_preproc:
            st.markdown("### Stage 1: Preprocessing & Offset Mapping")
            st.markdown(f"**Normalized Text:** `{result.preprocessing.normalized_text}`")
            st.markdown(
                f"**Sentence Count:** {len(result.preprocessing.sentences)} "
                f"| **Token Count:** {result.preprocessing.metadata.get('token_count', 0)}"
            )

            sent_rows = []
            for s in result.preprocessing.sentences:
                sent_rows.append({
                    "Index": s.sentence_id,
                    "Raw Start": s.raw_start,
                    "Raw End": s.raw_end,
                    "Sentence Text": s.text,
                    "Token Count": len(s.tokens),
                })
            st.dataframe(pd.DataFrame(sent_rows), width='stretch', hide_index=True)

            if result.preprocessing.spell_flags:
                st.warning(f"Medical Spell Flags Detected: {result.preprocessing.spell_flags}")

        # --- TAB 8: Patient JSON ---
        with tab_json:
            st.markdown("### Stage 7: Structured PatientRecord (JSON)")
            st.json(result.patient_record.model_dump())

        # --- TAB 9: Physician Review & Audit ---
        with tab_review:
            st.markdown("### Stage 9: Physician Review Station")
            st.write(
                "Allows healthcare professionals to ACCEPT, EDIT, REJECT, or VERIFY "
                "extracted findings with audit logging."
            )

            col_rev_form, col_rev_log = st.columns([1, 1])

            with col_rev_form:
                reviewer_name = st.text_input(
                    "Reviewer Name / ID:", value="Dr. Physician, MD"
                )
                review_item_options = {
                    e.id: f"{e.span.text} ({e.type.value} at {e.span.start}:{e.span.end})"
                    for e in result.entities
                }

                if review_item_options:
                    selected_item_id = st.selectbox(
                        "Select Item to Review:",
                        options=list(review_item_options.keys()),
                        format_func=lambda x: review_item_options[x],
                    )

                    review_note = st.text_input(
                        "Clinical Note / Rationale (optional):", value=""
                    )
                    edit_new_text = st.text_input(
                        "New Value (required only for EDIT action):", value=""
                    )

                    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

                    with btn_col1:
                        if st.button("✅ Accept", width='stretch'):
                            _log_review_action(
                                "ACCEPT",
                                reviewer_name,
                                review_item_options[selected_item_id],
                                review_note or "Finding accepted as accurate",
                            )
                            st.success(f"Accepted {review_item_options[selected_item_id]}")

                    with btn_col2:
                        if st.button("✏️ Edit", width='stretch'):
                            if edit_new_text:
                                _log_review_action(
                                    "EDIT",
                                    reviewer_name,
                                    review_item_options[selected_item_id],
                                    review_note or f"Edited to '{edit_new_text}'",
                                    new_value=edit_new_text,
                                )
                                st.success(f"Edited finding to '{edit_new_text}'")
                            else:
                                st.error("Please provide a new value for the EDIT action.")

                    with btn_col3:
                        if st.button("❌ Reject", width='stretch'):
                            _log_review_action(
                                "REJECT",
                                reviewer_name,
                                review_item_options[selected_item_id],
                                review_note or "Rejected by clinician",
                            )
                            st.warning(f"Rejected {review_item_options[selected_item_id]}")

                    with btn_col4:
                        if st.button("🔒 Verify", width='stretch'):
                            _log_review_action(
                                "VERIFY",
                                reviewer_name,
                                review_item_options[selected_item_id],
                                review_note or "Verified against source narrative",
                            )
                            st.info(f"Verified {review_item_options[selected_item_id]}")
                else:
                    st.info("No items available to review.")

            with col_rev_log:
                st.markdown("#### Audit Trail Log")
                if st.session_state.review_audit_log:
                    df_audit = pd.DataFrame(st.session_state.review_audit_log)
                    st.dataframe(df_audit, width='stretch', hide_index=True)

                    st.download_button(
                        "📥 Export Audit Log (.json)",
                        data=json.dumps(st.session_state.review_audit_log, indent=2),
                        file_name="physician_review_audit.json",
                        mime="application/json",
                        width='stretch',
                    )
                else:
                    st.info("No review actions recorded in this session yet.")


# ---------------------------------------------------------------------------
# Voice Input Section (English Only)
# ---------------------------------------------------------------------------
def _render_voice_input_section(current_text: str):
    """Render the voice input microphone button and result for English mode."""
    st.markdown(
        '<div class="voice-note">'
        '🎙️ <b>Voice Input (English Only)</b> -- Click the button below to record. '
        'Speak clearly in English. Internet connection required (Google STT).'
        '</div>',
        unsafe_allow_html=True,
    )

    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        mic_available = is_microphone_available()

        if not mic_available:
            st.button(
                "🎙️ Record Voice",
                disabled=True,
                help="No microphone detected on this system.",
                width='stretch',
            )
            st.caption("No microphone found.")
        else:
            if st.button("🎙️ Record Voice (English)", width='stretch', type="secondary"):
                with st.spinner(
                    "Listening... Speak clearly in English (up to 30 seconds)."
                ):
                    voice_result: VoiceInputResult = transcribe_english_audio(
                        timeout_seconds=8, phrase_time_limit=30
                    )

                if voice_result.success:
                    st.session_state.voice_transcribed_text = voice_result.transcribed_text
                    st.session_state.selected_case_text = voice_result.transcribed_text
                    st.session_state.pipeline_result = None
                    st.success("Voice transcribed successfully!")
                    st.rerun()
                else:
                    st.error(f"Voice input failed: {voice_result.error}")
                    st.session_state.voice_transcribed_text = ""

    with col_v2:
        if st.session_state.voice_transcribed_text:
            st.markdown("**Last voice transcription:**")
            st.markdown(
                f'<div class="translated-box">'
                f'{st.session_state.voice_transcribed_text}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Clear Voice Text", width='stretch'):
                st.session_state.voice_transcribed_text = ""
                st.rerun()


# ---------------------------------------------------------------------------
# Review Action Logger
# ---------------------------------------------------------------------------
def _log_review_action(
    action: str,
    reviewer: str,
    item: str,
    note: str,
    new_value: Optional[str] = None,
):
    record = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "action": action,
        "reviewer": reviewer,
        "item": item,
        "note": note,
    }
    if new_value:
        record["new_value"] = new_value
    st.session_state.review_audit_log.append(record)


# ---------------------------------------------------------------------------
# Pipeline Architecture Visualization
# ---------------------------------------------------------------------------
def render_pipeline_architecture():
    """
    Render an interactive pipeline architecture diagram.
    Graphify-style knowledge graph showing module relationships,
    data flow, schemas, and extension points.
    """
    st.subheader("🗺️ Pipeline Architecture & Knowledge Graph")
    st.markdown(
        "Visual overview of the 9-stage clinical NLP pipeline, module relationships, "
        "shared schema contracts, and multilingual/voice pre-processing layers."
    )

    # ------------------------------------------------------------------
    # Tab layout: Overview | Data Flow | Module Map | Schema Contracts
    # ------------------------------------------------------------------
    tab_overview, tab_flow, tab_modules, tab_schema, tab_lang = st.tabs([
        "🔭 System Overview",
        "⚙️ Data Flow (9 Stages)",
        "📦 Module Map",
        "📐 Schema Contracts",
        "🌐 Multilingual & Voice Layer",
    ])

    # ---- OVERVIEW ----
    with tab_overview:
        st.markdown("""
### AI Clinical Patient Case-Taking System -- Architecture Overview

This system converts **unstructured patient narrations** (plain text, optionally in Hindi or Kannada)
into **structured, physician-ready clinical records** through a deterministic 9-stage NLP pipeline.

#### Design Principles
| Principle | Implementation |
|---|---|
| **Modularity** | Each stage is a separate Python class with a `run()` method |
| **Provenance** | Every span carries `(start, end)` offsets into the ORIGINAL raw text |
| **Determinism** | Rule-based logic used first; ML is an optional plug-in |
| **Explainability** | All assertion triggers, temporal links, and relations are explicitly recorded |
| **Separation of Concerns** | Orchestrator only passes data between stages; no NLP logic in pipeline.py |
| **Decision Support Only** | No autonomous diagnosis or treatment recommendations |

#### Technology Stack
| Layer | Tools |
|---|---|
| NLP Engine | Custom rule-based + regex (spaCy/HF optional extensions) |
| Schemas | Pydantic v2 |
| Summarization | Template-based (LLM optional extension) |
| Translation | deep-translator (Google Translate API) |
| Voice STT | SpeechRecognition + pyaudio (Google Web Speech) |
| Web UI | Streamlit 1.x |
| Evaluation | Custom F1/Precision/Recall evaluator (20 synthetic cases) |
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Pre-Pipeline Input Layer")
            for item in [
                "🎙️ Voice Input (English only) → SpeechRecognition",
                "🌐 Hindi text → deep-translator → English",
                "🌐 Kannada text → deep-translator → English",
                "🇬🇧 English text → passed directly",
            ]:
                st.markdown(f"- {item}")

        with col_b:
            st.markdown("#### Post-Pipeline Output Layer")
            for item in [
                "📋 Physician-ready Clinical Summary",
                "📦 Structured PatientRecord JSON",
                "🔗 Relation graph (medication-dosage, symptom-duration, ...)",
                "🩺 Physician Review + Audit Trail",
                "📊 Benchmark Evaluation (F1/P/R vs ground truth)",
            ]:
                st.markdown(f"- {item}")

    # ---- DATA FLOW ----
    with tab_flow:
        st.markdown("### 9-Stage Sequential Data Flow")
        st.markdown(
            "Each stage consumes the output of the previous stage and produces "
            "Pydantic-validated structured objects."
        )

        stages = [
            ("0", "🌐 Multilingual Input Layer",
             "Voice STT (EN) / Hindi/Kannada text → English translation",
             "#F0FDF4", "#166534"),
            ("1", "🧹 Preprocessing",
             "Cleaning, contraction expansion, sentence splitting, tokenization, offset mapping",
             "#EFF6FF", "#1E40AF"),
            ("2", "🏷️ NER (Named Entity Recognition)",
             "14 entity types: SYMPTOM, DISEASE, MEDICATION, DOSAGE, FREQUENCY, TEST, ...",
             "#EFF6FF", "#1E40AF"),
            ("3", "⚖️ Assertion / Negation",
             "NegEx-inspired: PRESENT / NEGATED / POSSIBLE / HISTORICAL / FAMILY per entity",
             "#EFF6FF", "#1E40AF"),
            ("4", "⏱️ Temporal Analysis",
             "DURATION, ONSET, relative/absolute dates, timeline construction, anaphora resolution",
             "#EFF6FF", "#1E40AF"),
            ("5", "🔗 Relation Extraction",
             "SYMPTOM-DURATION, MEDICATION-DOSAGE, TEST-RESULT, PROCEDURE-DATE, ...",
             "#EFF6FF", "#1E40AF"),
            ("6", "📑 Section Classification",
             "Chief Complaint / HPI / Past History / Drug History / Allergy / Family / Personal / ROS",
             "#EFF6FF", "#1E40AF"),
            ("7", "📦 Patient Representation",
             "Assembles structured PatientRecord JSON with full provenance",
             "#FFF7ED", "#92400E"),
            ("8", "📋 Summarization",
             "Template-based physician-ready summary from PatientRecord (never from raw text)",
             "#FFF7ED", "#92400E"),
            ("9", "🩺 Physician Review",
             "ACCEPT / EDIT / REJECT / VERIFY with real-time audit log",
             "#FEF2F2", "#991B1B"),
        ]

        for num, name, desc, bg, fg in stages:
            st.markdown(
                f'<div style="background:{bg};border-left:4px solid {fg};'
                f'padding:10px 16px;border-radius:6px;margin-bottom:8px;">'
                f'<span style="color:{fg};font-weight:700;">Stage {num}: {name}</span>'
                f'<br/><span style="color:#374151;font-size:0.88rem;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if num != "9":
                st.markdown(
                    '<div style="text-align:center;color:#9CA3AF;font-size:1.2rem;'
                    'margin:2px 0;">↓</div>',
                    unsafe_allow_html=True,
                )

    # ---- MODULE MAP ----
    with tab_modules:
        st.markdown("### Module Map: Files & Responsibilities")

        module_data = [
            ("src/common/schemas.py", "Shared Schemas", "Pydantic models: Span, Entity, Assertion, TimeExpr, Relation, PatientRecord"),
            ("src/common/translator.py", "Translation Utility", "Hindi/Kannada → English via deep-translator (pre-pipeline layer)"),
            ("src/common/voice_input.py", "Voice STT Utility", "English speech → text via SpeechRecognition + Google STT"),
            ("src/preprocessing/preprocessor.py", "Stage 1: Preprocessing", "Contraction expansion, offset mapping, sentence splitting, normalization"),
            ("src/ner/ner_engine.py", "Stage 2: NER", "Dictionary + regex NER for 14 clinical entity types"),
            ("src/ner/lexicons.py", "NER Lexicons", "SYMPTOM, DISEASE, MEDICATION, BODY_PART, TEST, ALLERGEN, ... frozensets"),
            ("src/negation/assertion_engine.py", "Stage 3: Assertion", "NegEx-inspired context window assertion classification"),
            ("src/negation/triggers.py", "Negation Triggers", "NEGATION / POSSIBLE / HISTORICAL / FAMILY trigger word sets"),
            ("src/temporal/temporal_engine.py", "Stage 4: Temporal", "Duration, onset, relative date extraction + timeline builder"),
            ("src/temporal/patterns.py", "Temporal Patterns", "Regex patterns for date/duration/onset expressions"),
            ("src/relation_extraction/relation_engine.py", "Stage 5: Relations", "Intra-sentence candidate pairs + rule-based relation classifier"),
            ("src/relation_extraction/rules.py", "Relation Rules", "SYMPTOM-DURATION, MEDICATION-DOSAGE, TEST-RESULT, ... rules"),
            ("src/section_classification/section_classifier.py", "Stage 6: Sections", "Deterministic rule-based section assignment (8 sections)"),
            ("src/patient_representation/patient_builder.py", "Stage 7: Patient Record", "Assembles PatientRecord from all upstream stage outputs"),
            ("src/summarization/summarizer.py", "Stage 8: Summarization", "Template-based physician summary from PatientRecord only"),
            ("src/review/review_engine.py", "Stage 9: Review", "ACCEPT/EDIT/REJECT/VERIFY physician review + audit log"),
            ("src/pipeline/clinical_pipeline.py", "Orchestrator", "Executes stages 1-9 in order; no NLP logic"),
            ("main.py", "CLI Entry Point", "--text / --case / --file / --evaluate / --legacy modes"),
            ("app/streamlit_app.py", "Web UI", "Streamlit app with multilingual input, voice STT, 9 visualization tabs"),
            ("evaluation/case_evaluator.py", "Evaluator", "F1/Precision/Recall against 20 synthetic ground-truth cases"),
            ("data/sample/", "Synthetic Cases", "20 synthetic patient narration cases + ground_truth.json"),
        ]

        df_mods = pd.DataFrame(module_data, columns=["File / Module", "Role", "Description"])
        st.dataframe(df_mods, width='stretch', hide_index=True)

    # ---- SCHEMA CONTRACTS ----
    with tab_schema:
        st.markdown("### Shared Schema Contracts (`src/common/schemas.py`)")
        st.markdown(
            "All pipeline stages consume and produce these Pydantic models. "
            "Every span retains character offsets into the ORIGINAL raw text."
        )

        schema_info = {
            "Span": {
                "fields": ["text: str", "start: int (raw offset)", "end: int (raw offset)", "source_sentence_id: int"],
                "purpose": "Character-level anchor into original raw text. Never loses provenance.",
            },
            "Entity": {
                "fields": ["id: str (UUID)", "span: Span", "type: EntityType", "confidence: float"],
                "purpose": "A detected clinical named entity with type and provenance.",
            },
            "Assertion": {
                "fields": ["entity_id: str", "status: AssertionStatus", "experiencer: Experiencer", "trigger_text: Optional[str]", "confidence: float"],
                "purpose": "Assertion context for one entity: PRESENT/NEGATED/POSSIBLE/HISTORICAL/FAMILY.",
            },
            "TimeExpr": {
                "fields": ["id: str", "span: Span", "type: TemporalType", "raw_value: str", "normalized_value: Optional[str]", "linked_entity_ids: List[str]"],
                "purpose": "A temporal expression with normalized value and entity linkages.",
            },
            "Relation": {
                "fields": ["id: str", "head_entity_id: str", "tail_entity_id: str", "relation_type: RelationType", "source_sentence_id: int", "evidence_text: Optional[str]"],
                "purpose": "A directed clinical relation between two entities.",
            },
            "PatientRecord": {
                "fields": ["id: str", "sections: Dict[str, List]", "entities: List[Entity]", "assertions: List[Assertion]", "time_expressions: List[TimeExpr]", "relations: List[Relation]", "timeline: List[TimelineNode]"],
                "purpose": "The complete structured patient representation -- source of truth for summarization.",
            },
        }

        for schema_name, info in schema_info.items():
            with st.expander(f"📐 `{schema_name}`", expanded=False):
                st.markdown(f"**Purpose:** {info['purpose']}")
                st.markdown("**Fields:**")
                for f in info["fields"]:
                    st.markdown(f"  - `{f}`")

    # ---- MULTILINGUAL & VOICE LAYER ----
    with tab_lang:
        st.markdown("### Multilingual & Voice Input Architecture")

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            st.markdown("#### Text Input Languages")
            lang_data = [
                ("English", "en", "Direct to pipeline", "Full (text + voice)"),
                ("Hindi (हिंदी)", "hi", "deep-translator → English", "Text only"),
                ("Kannada (ಕನ್ನಡ)", "kn", "deep-translator → English", "Text only"),
            ]
            df_lang = pd.DataFrame(
                lang_data,
                columns=["Language", "Code", "Pipeline Path", "Input Modes"],
            )
            st.dataframe(df_lang, width='stretch', hide_index=True)

            st.markdown("""
**Translation Flow:**
```
Hindi / Kannada text
        ↓
deep-translator (GoogleTranslator)
    source=hi/kn → target=en
        ↓
English text
        ↓
9-stage NLP pipeline
```
Chunked at 4000 chars to avoid API limits.
Original text + source language code retained for provenance.
            """)

        with col_l2:
            st.markdown("#### Voice Input (English Only)")
            st.markdown("""
**Voice Pipeline:**
```
Microphone
    ↓
SpeechRecognition (pyaudio)
    Ambient noise calibration (1s)
    Listen with timeout=8s
    phrase_time_limit=30s
        ↓
Google Web Speech API
    language=en-IN
        ↓
Transcribed English text
        ↓
9-stage NLP pipeline
```
**Constraints:**
- English ONLY (system requirement)
- Requires internet connection
- Requires microphone (pyaudio)
- Energy threshold: 300 (auto-adjusting)
- Graceful fallback on mic-not-found / API errors
            """)

        st.divider()
        st.markdown("#### Extension Points")
        ext_data = [
            ("Voice -- Other Languages", "Add `whisper` (offline) with lang param; set language='hi' or 'kn'", "Out of scope (v1)"),
            ("Translation -- Offline", "Replace GoogleTranslator with `argostranslate` or `Helsinki-NLP` HF models", "Optional future"),
            ("NER -- ML Model", "Swap `ClinicalNER` with `HFNERAdapter` using fine-tuned bert-base-NER", "Optional future"),
            ("Summarization -- LLM", "Add LLM backend to `Summarizer` with structured-record grounding check", "Optional future"),
        ]
        df_ext = pd.DataFrame(
            ext_data,
            columns=["Feature", "How to Extend", "Status"],
        )
        st.dataframe(df_ext, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Benchmark Evaluation
# ---------------------------------------------------------------------------
def render_benchmark_evaluation():
    """Render the benchmark evaluation view computing Precision, Recall, and F1."""
    st.subheader("📊 Benchmark Evaluation: 20 Clinical Synthetic Cases")
    st.write(
        "Evaluates the 9-stage pipeline against ground truth annotations "
        "(`data/sample/ground_truth.json`)."
    )

    evaluator = CaseEvaluator()

    if st.button("⚡ Run Full Benchmark Evaluation", type="primary"):
        with st.spinner("Evaluating 20 clinical benchmark cases..."):
            results = evaluator.evaluate_all()

        if "error" in results:
            st.error(results["error"])
            return

        o_ner = results.get("ner", {})
        o_ass = results.get("assertion", {})
        o_rel = results.get("relation", {})

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(
                "NER F1 Score",
                f"{o_ner.get('f1', 0):.4f}",
                f"Precision: {o_ner.get('precision', 0):.2f}",
            )
        with col_m2:
            st.metric(
                "Assertion F1 Score",
                f"{o_ass.get('f1', 0):.4f}",
                f"Precision: {o_ass.get('precision', 0):.2f}",
            )
        with col_m3:
            st.metric(
                "Relation F1 Score",
                f"{o_rel.get('f1', 0):.4f}",
                f"Precision: {o_rel.get('precision', 0):.2f}",
            )

        st.markdown("#### Detailed Case-by-Case Benchmark Results")
        case_rows = []
        for r in results.get("case_results", []):
            case_rows.append({
                "Case": r["case"],
                "NER Precision": r["ner"]["precision"],
                "NER Recall": r["ner"]["recall"],
                "NER F1": r["ner"]["f1"],
                "Assertion Precision": r["assertion"]["precision"],
                "Assertion Recall": r["assertion"]["recall"],
                "Assertion F1": r["assertion"]["f1"],
                "Relation F1": r["relation"]["f1"],
            })
        st.dataframe(pd.DataFrame(case_rows), width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Legacy Analyzer
# ---------------------------------------------------------------------------
def render_legacy_analyzer():
    """Render the legacy document-level processing flow."""
    st.subheader("📁 Legacy Document Summarizer")
    st.info(
        "Provides file extraction and summarization for PDF, DOCX, and TXT documents."
    )
    uploaded_file = st.file_uploader(
        "Upload a medical file (.txt, .pdf, .docx):", type=["txt", "pdf", "docx"]
    )
    if uploaded_file:
        content = uploaded_file.read()
        if uploaded_file.name.endswith(".txt"):
            text = content.decode("utf-8", errors="replace")
            st.text_area("Extracted Text Preview:", value=text[:1000], height=200)
            if st.button("Process Document with Clinical Pipeline"):
                st.session_state.selected_case_text = text
                st.session_state.pipeline_result = None
                st.session_state.translation_result = None
                st.session_state.input_language = "English"
                st.success(
                    "Document transferred to Case-Taking Pipeline! "
                    "Switch to 'Case-Taking Pipeline' mode in sidebar."
                )


if __name__ == "__main__":
    main()

