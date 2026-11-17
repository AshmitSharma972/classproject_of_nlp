"""
Report Generator Module: Compiles extracted clinical entities, key points, summaries,
and source metadata into standardized structured patient summary reports.
"""

from typing import Dict, Any, List
import json
from config import MEDICAL_DISCLAIMER
from src.utils import get_logger

logger = get_logger("ReportGenerator")


class ReportGenerator:
    """Formats clinical NLP outputs into structured patient summary documents."""

    def build_structured_summary(
        self,
        structured_slots: Dict[str, List[str]],
        key_points: List[Dict[str, Any]],
        summary_info: Dict[str, Any],
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assembles full report object conforming strictly to prompt specifications.
        """
        report_data = {
            "title": "PATIENT SUMMARY",
            "chief_complaints": structured_slots.get("Chief Complaints", []),
            "symptoms": structured_slots.get("Symptoms", []),
            "medical_history": structured_slots.get("Medical History", []),
            "medications": structured_slots.get("Medications", []),
            "allergies": structured_slots.get("Allergies", []),
            "tests_and_results": structured_slots.get("Tests and Results", []),
            "important_findings": structured_slots.get("Important Findings", []) + structured_slots.get("Negated Findings", []),
            "relevant_dates_duration": structured_slots.get("Relevant Dates/Duration", []),
            "key_points": [kp["sentence"] for kp in key_points],
            "ai_generated_summary": summary_info.get("summary_text", ""),
            "summary_mode": summary_info.get("mode", ""),
            "source_information": {
                "source_name": doc_metadata.get("source_name", "Unknown Document"),
                "file_type": doc_metadata.get("file_type", "unknown"),
                "page_count": doc_metadata.get("page_count", 1),
                "is_scanned": doc_metadata.get("is_scanned", False)
            },
            "disclaimer": MEDICAL_DISCLAIMER
        }

        return report_data

    def generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Renders the structured report data as clean GitHub-style Markdown."""
        lines = []

        lines.append("# PATIENT SUMMARY\n")

        def format_section(title: str, items: List[str]):
            lines.append(f"### {title}:")
            if items:
                for item in items:
                    lines.append(f"* {item}")
            else:
                lines.append("* Not mentioned in the document.")
            lines.append("")

        format_section("Chief Complaints", report_data["chief_complaints"])
        format_section("Symptoms", report_data["symptoms"])
        format_section("Medical History", report_data["medical_history"])
        format_section("Medications", report_data["medications"])
        format_section("Allergies", report_data["allergies"])
        format_section("Tests and Results", report_data["tests_and_results"])
        format_section("Important Findings", report_data["important_findings"])
        format_section("Relevant Dates/Duration", report_data["relevant_dates_duration"])
        format_section("Key Points", report_data["key_points"])

        lines.append("### AI-Generated Summary:")
        lines.append(f"> {report_data['ai_generated_summary']}\n")

        lines.append("### Source Information:")
        src = report_data["source_information"]
        lines.append(f"* **Document Name**: {src['source_name']}")
        lines.append(f"* **File Type**: {src['file_type'].upper()}")
        lines.append(f"* **Page / Paragraph Count**: {src['page_count']}")
        lines.append(f"* **Scanned Document**: {'Yes (OCR required)' if src['is_scanned'] else 'No (Direct Text)'}")
        lines.append("")

        lines.append("---")
        lines.append(f"**NOTICE**: {report_data['disclaimer']}")

        return "\n".join(lines)

    def generate_plain_text_report(self, report_data: Dict[str, Any]) -> str:
        """Renders structured report data as plain text."""
        md = self.generate_markdown_report(report_data)
        # Strip markdown hashes and quotes for plain text representation
        plain = md.replace("# ", "").replace("### ", "").replace("> ", "")
        return plain

    def generate_json_report(self, report_data: Dict[str, Any]) -> str:
        """Renders report data as formatted JSON string."""
        return json.dumps(report_data, indent=2)
