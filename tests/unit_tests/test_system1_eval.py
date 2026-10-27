import json
import runpy
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evaluation():
    return runpy.run_path(str(ROOT / "scripts/system1_eval.py"))


def test_corpus_splits_and_label_separation(evaluation):
    cases, digest = evaluation["load_corpus"]()
    assert len(digest) == 64
    assert len(cases) == 36
    ids = []
    for split in ("train", "development", "heldout"):
        subset = [c for c in cases if c.split == split]
        assert len(subset) == 12
        assert {
            label: sum(c.expected == label for c in subset)
            for label in ("fast", "powerful", "allow", "block")
        } == dict.fromkeys(("fast", "powerful", "allow", "block"), 3)
        ids.extend(c.id for c in subset)
        assert all("expected" not in c.evidence and "label_reason" not in c.evidence for c in subset)
    assert len(ids) == len(set(ids))
    selected = evaluation["select_cases"](cases, "development", 4, 12)
    assert [c.id for c in selected] == [c.id for c in evaluation["select_cases"](cases, "development", 4, 12)]
    with pytest.raises(ValueError, match="budget"):
        evaluation["select_cases"](cases, "development", 4, 11)
    with pytest.raises(ValueError, match="Training"):
        evaluation["select_cases"](cases, "train", 4, 12)


def test_holdout_requires_frozen_successful_development(evaluation, tmp_path):
    config = evaluation["EvaluationConfig"]()
    with pytest.raises(ValueError, match="requires"):
        evaluation["freeze_reference"](None, "heldout", config, "hash")
    report = {
        "protocol": evaluation["PROTOCOL"],
        "split": "development",
        "status": "complete",
        "config": config.model_dump(),
        "corpus_sha256": "hash",
        "implementation_sha256": evaluation["implementation_hash"](),
        "versions": evaluation["versions"](),
        "summary": {"failed_requests": 0},
    }
    path = tmp_path / "development.json"
    path.write_text(json.dumps(report))
    assert len(evaluation["freeze_reference"](path, "heldout", config, "hash")) == 64
    for key, bad in [
        ("config", {}),
        ("corpus_sha256", "other"),
        ("implementation_sha256", "other"),
        ("versions", {}),
        ("status", "running"),
        ("summary", {"failed_requests": 1}),
        ("split", "heldout"),
    ]:
        path.write_text(json.dumps({**report, key: bad}))
        with pytest.raises(ValueError, match="reference"):
            evaluation["freeze_reference"](path, "heldout", config, "hash")


def record(task, expected, predicted, probability=None, status="ok"):
    return {
        "task": task,
        "expected": expected,
        "predicted": predicted,
        "probability": probability,
        "status": status,
        "fallback": status != "ok",
        "latency_seconds": 0.1,
        "input_tokens": 100,
    }


def test_metrics_include_failures_and_reference_baseline(evaluation):
    rows = [
        record("route", "fast", "powerful", status="failed"),
        record("route", "powerful", "powerful"),
        record("screen", "block", "allow", 0.1),
        record("screen", "block", "block", 0.8),
        record("screen", "allow", "block", status="failed"),
        record("screen", "allow", "allow", 0.2),
    ]
    summary = evaluation["summarize"](rows, evaluation["EvaluationConfig"]())
    assert summary["evaluated_cases"] == 6 and summary["failed_requests"] == 2
    route = summary["routing"]
    guard = summary["screening"]
    assert route["accuracy_including_fallback"]["rate"] == 0.5
    assert route["fixed_powerful_accuracy"]["rate"] == 0.5
    assert guard["unsafe_allows"] == guard["false_blocks"] == 1
    assert guard["precision"]["rate"] == guard["recall"]["rate"] == 0.5
    assert guard["brier_on_successful_responses"] == pytest.approx((0.81 + 0.04 + 0.04) / 3)
    assert sum(b["count"] for b in guard["calibration_bins"]) == 3
    assert not summary["cost_complete"]
    assert summary["estimated_known_decision_cost_usd"] == pytest.approx(400 * 0.035 / 1_000_000)
    assert route["host_task_success"] is None and route["host_cost_savings_usd"] is None
    assert not summary["production_readiness_demonstrated"]
    assert summary["routing"]["accuracy_including_fallback"]["wilson_95"][0] < 0.5
    assert evaluation["wilson"](0, 0) is None
    assert evaluation["percentile"]([1, 2, 3, 4], 0.95) == 4


def test_runner_paces_starts_and_checkpoints_failed_cases(evaluation):
    cases, _ = evaluation["load_corpus"]()
    cases = cases[:3]
    now = [0.0]
    starts = []
    snapshots = []

    def sleep(seconds):
        now[0] += seconds

    def execute(case):
        starts.append(now[0])
        now[0] += 0.01
        if len(starts) == 2:
            raise RuntimeError("private raw response must not appear")
        return {
            "status": "ok",
            "predicted": case.expected,
            "fallback": False,
            "probability": None,
            "input_tokens": 1,
        }

    records = evaluation["run_cases"](
        cases,
        evaluation["EvaluationConfig"](),
        execute,
        lambda rows: snapshots.append(json.loads(json.dumps(rows))),
        clock=lambda: now[0],
        sleep=sleep,
    )
    assert starts == pytest.approx([0, 0.25, 0.5])
    assert [len(s) for s in snapshots] == [1, 2, 3]
    assert records[1]["status"] == "failed" and records[1]["predicted"] == "powerful"
    assert "private raw response" not in json.dumps(records)


def provider_response(request):
    data = json.loads(request.content)
    if "model_route" in data["questions"]:
        answers = {
            "model_route": {
                "type": "choice",
                "choice": "powerful",
                "confidence": 0.8,
                "probabilities": {"fast": 0.1, "powerful": 0.9},
            }
        }
    else:
        answers = {"tool_risk": {"type": "noul", "noul": 0.8}}
    return httpx.Response(
        200,
        json={
            "id": "dec_eval",
            "model": data["model"],
            "answers": answers,
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "request_id": "req_eval",
            "evaluation_time_ms": 1,
        },
    )


def test_actual_middleware_requests_contain_context_not_labels(evaluation):
    cases, _ = evaluation["load_corpus"]()
    observed = []

    def handle(request):
        observed.append(json.loads(request.content))
        return provider_response(request)

    with evaluation["RecordingClassifier"](
        api_key="test", transport=httpx.MockTransport(handle), max_attempts=1
    ) as classifier:
        for case in cases:
            result = evaluation["evaluate_case"](case, classifier, evaluation["EvaluationConfig"]())
            assert result["status"] == "ok" and result["request_id"] == "req_eval"
            body = observed[-1]
            assert case.label_reason not in json.dumps(body)
            assert "base_rate" not in json.dumps(body)
            if case.task == "route":
                assert len(body["data"]["messages"]) == len(case.evidence["messages"])
            else:
                assert body["data"]["tool_definition"]["description"] == case.evidence["tool"]["description"]
    assert len(observed) == 36


def test_cli_requires_explicit_opt_in(evaluation, tmp_path):
    with pytest.raises(SystemExit):
        evaluation["main"](["--split", "development", "--output", str(tmp_path / "result.json")])
    assert not (tmp_path / "result.json").exists()


def test_cli_counts_real_transport_attempts_and_refuses_overwrite(evaluation, monkeypatch, tmp_path):
    monkeypatch.setenv("FINIC_EVALUATION_TESTS", "1")
    monkeypatch.setenv("FINIC_API_KEY", "test")
    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(provider_response), **kwargs)
    )
    # Only remove wall-clock pacing in this CLI test; pacing itself is checked above.
    original_run = evaluation["run_cases"]
    monkeypatch.setitem(
        evaluation["main"].__globals__,
        "run_cases",
        lambda cases, config, execute, checkpoint: original_run(
            cases, config, execute, checkpoint, sleep=lambda _: None
        ),
    )
    path = tmp_path / "development.json"
    assert evaluation["main"](["--split", "development", "--output", str(path)]) == 0
    report = json.loads(path.read_text())
    assert report["http_attempts"] == 12
    assert sum(r["http_attempts"] for r in report["records"]) == 12
    assert report["status"] == "complete" and report["summary"]["failed_requests"] == 0
    assert all(r["calibrated"] for r in report["records"])
    assert "Authorization" not in path.read_text()
    with pytest.raises(FileExistsError):
        evaluation["main"](["--split", "development", "--output", str(path)])


def test_provider_failure_keeps_request_id_without_stale_probability(evaluation):
    cases, _ = evaluation["load_corpus"]()
    case = next(c for c in cases if c.task == "screen")
    responses = iter(
        [
            None,
            httpx.Response(
                503,
                headers={"x-request-id": "req_failed"},
                json={"error": {"type": "overloaded", "message": "private"}},
            ),
        ]
    )

    def handle(request):
        result = next(responses)
        return result if result is not None else provider_response(request)

    with evaluation["RecordingClassifier"](
        api_key="test", max_attempts=1, transport=httpx.MockTransport(handle)
    ) as classifier:
        first = evaluation["evaluate_case"](case, classifier, evaluation["EvaluationConfig"]())
        second = evaluation["evaluate_case"](case, classifier, evaluation["EvaluationConfig"]())
    assert first["status"] == "ok"
    assert second["status"] == "failed" and second["predicted"] == "block"
    assert second["probability"] is None and second["input_tokens"] == 0
    assert second["request_id"] == "req_failed" and second["provider_error_type"] == "overloaded"
    assert "private" not in json.dumps(second)


def test_reference_baseline_tracks_corpus(evaluation):
    _, digest = evaluation["load_corpus"]()
    baseline = json.loads((ROOT / "tests/evaluation/system1-v1/reference-baseline.json").read_text())
    assert baseline["corpus_sha256"] == digest
    assert baseline["http_attempts"] == 0
    assert all(
        split["fixed_powerful_routing_accuracy"]["rate"] == 0.5 for split in baseline["results"].values()
    )
