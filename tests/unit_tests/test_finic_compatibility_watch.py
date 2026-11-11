"""Contract tests for the provider compatibility monitor."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "finic_compatibility", Path(__file__).resolve().parents[2] / "scripts/finic_compatibility.py"
)
watch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = watch
SPEC.loader.exec_module(watch)


def reference(extra=""):
    rows = [
        "| [`score`](/reference/score) | `POST /v1/score` | Score |",
        "| [`ask`](/reference/ask) | `POST /v1/ask` | Ask |",
        "| [`job`](/reference/job) | `GET /v1/jobs/{job_id}` | Job |",
        "| [`models`](/reference/models) | `GET /v1/scoring_models` | List |",
        "| [`create`](/reference/create) | `POST /v1/scoring_models` | Create |",
    ]
    return "# API reference (/reference)\n" + "\n".join(rows) + extra + "\n## Errors\n"


def test_reference_parses_and_detects_format_drift():
    assert len(watch.parse_reference(reference())) == 5
    with pytest.raises(ValueError, match="could not be parsed"):
        watch.parse_reference(reference("\n| [`new`](/reference/new) | `HEAD /v1/new` | New |"))
    with pytest.raises(ValueError, match="incomplete"):
        watch.parse_reference("# API reference\n## Errors\n")
    with pytest.raises(ValueError, match="structure"):
        watch.parse_reference("<html>maintenance</html>")


def test_assessment_catches_gaps_changes_new_and_removed():
    ops = {
        "POST /v1/score": watch.Operation("score", "POST", "/v1/score", "/reference/score", "new"),
        "POST /v1/ask": watch.Operation("ask", "POST", "/v1/ask", "/reference/ask", "x"),
        "GET /v1/new": watch.Operation("new", "GET", "/v1/new", "/reference/new", "y"),
    }
    base = {
        "operations": {
            "POST /v1/score": {"status": "supported", "digest": "old", "link": "/reference/score"},
            "POST /v1/ask": {"status": "partial", "digest": "x", "link": "/reference/ask"},
            "GET /v1/old": {"status": "supported", "digest": "z", "link": "/reference/old"},
        }
    }
    assert {(f["key"], f["reason"]) for f in watch.assess(ops, base)} == {
        ("POST /v1/score", "changed"),
        ("POST /v1/ask", "gap"),
        ("GET /v1/new", "new"),
        ("GET /v1/old", "removed"),
    }


def test_page_must_match_operation():
    op = watch.Operation("ask", "POST", "/v1/ask", "/reference/ask")
    digest = watch.page_digest(op, "# ask\n## POST https://api.finic.ai/v1/ask\n")
    assert len(digest) == 64
    with pytest.raises(ValueError, match="did not match"):
        watch.page_digest(op, "# Maintenance")


def test_issue_deduplication_respects_open_and_closed_evidence(monkeypatch):
    finding = {"key": "POST /v1/ask", "reason": "gap", "digest": "a" * 64, "link": "/reference/ask"}
    base = {"operations": {finding["key"]: {"note": "Review the alias."}}}
    stable, evidence = watch.marker(finding), watch.evidence_marker(finding)
    calls = []
    issues = []

    def run(args, **kwargs):
        calls.append(args)
        if args[2] == "list":
            return SimpleNamespace(stdout=json.dumps(issues))
        return SimpleNamespace(stdout="created")

    monkeypatch.setattr(subprocess, "run", run)
    watch.sync_issues([finding, finding], base)
    assert sum(args[2] == "create" for args in calls) == 1
    calls.clear()
    issues.append({"state": "OPEN", "body": stable})
    watch.sync_issues([finding], base)
    assert len(calls) == 1
    calls.clear()
    issues[:] = [{"state": "CLOSED", "body": f"{stable}\n{evidence}"}]
    watch.sync_issues([finding], base)
    assert len(calls) == 1
    calls.clear()
    issues[:] = [{"state": "CLOSED", "body": f"{stable}\nold evidence"}]
    watch.sync_issues([finding], base)
    assert len(calls) == 2


def test_provider_failure_is_not_reported_as_compatible(monkeypatch, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "operations": {
                    "GET /v1/a": {
                        "status": "supported",
                        "digest": "a" * 64,
                        "link": "/reference/a",
                        "note": "Reviewed",
                    }
                }
            }
        )
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(sys, "argv", ["watch", "--baseline", str(baseline)])
    monkeypatch.setattr(watch, "collect", lambda: (_ for _ in ()).throw(ValueError("unreadable")))
    assert watch.main() == 1
    assert "unknown" in summary.read_text().lower()
    assert "unknown" in capsys.readouterr().err


def test_invalid_baseline_never_claims_compatibility():
    with pytest.raises(ValueError, match="baseline"):
        watch.validate_baseline({"operations": {}})
    with pytest.raises(ValueError, match="status"):
        watch.validate_baseline({"operations": {"GET /v1/a": {"status": "typo"}}})
