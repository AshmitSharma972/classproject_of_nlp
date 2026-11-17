"""Benchmark page wiring and honest model/evaluation status."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import web_api
from scripts.evaluate_summarizer_checkpoint import score_summaries


def test_benchmark_endpoint_reports_gold_availability_and_checkpoint_status():
    report = __import__("evaluation.run_eval", fromlist=["run"]).run()
    assert report["clinical_pipeline_metrics"]["English"]["approved_cases"] == 0
    assert report["clinical_pipeline_metrics"]["English"]["metrics"] is None
    assert report["summary_model"]["active_in_live_pipeline"] is False
    assert report["summary_model"]["training_metrics"] is None
    summary_eval = report["summary_model"]["summary_evaluation"]
    if summary_eval is not None:
        assert summary_eval["samples_evaluated"] > 0
        assert {"rouge1_f1", "rouge2_f1", "rougeL_f1"} <= set(summary_eval["overall"])


def test_rouge_helper_scores_identical_synthetic_summary_as_full_overlap():
    metrics = score_summaries(["fever for five days"], ["fever for five days"])
    assert metrics == {
        "rouge1_precision": 100.0, "rouge1_recall": 100.0, "rouge1_f1": 100.0,
        "rouge2_precision": 100.0, "rouge2_recall": 100.0, "rouge2_f1": 100.0,
        "rougeL_precision": 100.0, "rougeL_recall": 100.0, "rougeL_f1": 100.0,
    }


def test_benchmark_page_loads_real_api_data_instead_of_fixed_unavailable_cards():
    html = open("index.html", encoding="utf-8").read()
    assert 'id="benchmark-results"' in html
    assert 'fetch("http://127.0.0.1:8765/api/benchmark")' in html
    assert 'id="benchmark-refresh"' in html
    benchmark_markup = html.split('id="benchmark-view"', 1)[1].split('id="architecture-view"', 1)[0]
    assert "clinical pipeline evaluation by language" not in benchmark_markup.lower()
    assert "Template-based" not in benchmark_markup
    assert "checkpoint status" not in benchmark_markup.lower()
    assert "ROUGE-1 F1" in html
    assert "ROUGE-2 F1" in html
    assert "ROUGE-L F1" in html
    assert "evaluation.per_language" not in html
    assert "Overall (" not in html
    assert "Not available</strong>" not in html


def test_benchmark_api_returns_json_with_request_id():
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_api.ClinIQHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/api/benchmark")
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["request_id"]
        assert payload["summary_model"]["active_path"] == "template"
        summary_eval = payload["summary_model"]["summary_evaluation"]
        if summary_eval is not None:
            assert summary_eval["samples_evaluated"] > 0
            assert {"en", "hi", "kn"} <= set(summary_eval["per_language"])
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
