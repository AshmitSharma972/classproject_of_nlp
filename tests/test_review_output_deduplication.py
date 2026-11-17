"""The review display must show a repeated visible finding only once."""
from pathlib import Path


def test_review_panel_uses_visible_review_items_deduplication():
    html = Path("index.html").read_text(encoding="utf-8")
    assert "function visibleReviewItems" in html
    assert "visibleReviewItems(patientRecord.section_items || [])" in html
    assert "visibleReviewItems(lastReportPayload.patient_record?.section_items || [])" in html
