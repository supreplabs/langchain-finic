"""Verify aggregate reporting without contacting Search Console."""

import importlib.util
from datetime import date
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "search_visibility", Path(__file__).resolve().parents[2] / "scripts/search_visibility.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_report_uses_expected_window_and_summarizes_api_results(monkeypatch):
    calls = []

    def fake(path, token, payload=None):
        calls.append((path, token, payload))
        if "searchAnalytics" in path:
            return {"rows": [{"clicks": 3, "impressions": 42}]}, None
        if "sitemaps" in path:
            return {"isPending": False, "errors": "0", "warnings": "0"}, None
        return {
            "inspectionResult": {"indexStatusResult": {"coverageState": "Indexed", "verdict": "PASS"}}
        }, None

    monkeypatch.setattr(module, "safe_call", fake)
    result = module.report("fake-token", date(2026, 11, 30))
    assert "2026-10-31 to 2026-11-27" in result
    assert "3 clicks; 42 impressions" in result
    assert "Homepage: Indexed" in result and "Host-model tutorial: Indexed" in result
    assert len(calls) == 4
    assert calls[0][2]["type"] == "web"
    assert calls[2][0] == "https://searchconsole.googleapis.com/v1/urlInspection/index:inspect"
    assert "fake-token" not in result


def test_unavailable_data_is_not_reported_as_zero(monkeypatch):
    monkeypatch.setattr(module, "safe_call", lambda *args, **kwargs: (None, "Google API request failed"))
    result = module.report("fake-token", date(2026, 11, 30))
    assert result.count("unavailable (Google API request failed)") == 4
    assert "0 clicks" not in result


def test_empty_search_result_does_not_claim_no_traffic(monkeypatch):
    monkeypatch.setattr(module, "safe_call", lambda *args, **kwargs: ({}, None))
    assert "no rows returned; metrics may be unavailable or zero" in module.report("fake-token", date.today())
