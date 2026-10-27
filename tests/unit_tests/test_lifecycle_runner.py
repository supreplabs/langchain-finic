import asyncio
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from langchain_finic import CreateScoreRequest, FinicClient, FinicError

ROOT = Path(__file__).resolve().parents[2]


def runner(optimize=0):
    path = ROOT / "scripts/live_lifecycle.py"
    namespace = {"__name__": "lifecycle_test", "__file__": str(path)}
    exec(compile(path.read_text(), str(path), "exec", optimize=optimize), namespace)
    return namespace


def test_synthetic_examples_meet_provider_minima():
    examples = runner()["synthetic_examples"]()
    assert len(CreateScoreRequest(data=examples).data) == 100
    assert sum(item["is_bad"] for item in examples) == 50


def test_runner_requires_separate_opt_in(monkeypatch, tmp_path):
    module = runner()
    monkeypatch.setenv("FINIC_API_KEY", "test-key")
    monkeypatch.setenv("FINIC_LIVE_TESTS", "1")
    with pytest.raises(RuntimeError, match="FINIC_LIFECYCLE_TESTS"):
        asyncio.run(module["run_live"](tmp_path / "evidence.json"))
    monkeypatch.setenv("FINIC_LIFECYCLE_TESTS", "1")
    monkeypatch.delenv("FINIC_API_KEY")
    with pytest.raises(RuntimeError, match="FINIC_API_KEY"):
        asyncio.run(module["run_live"](tmp_path / "evidence.json"))


def test_preserves_previous_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("FINIC_API_KEY", "test-key")
    monkeypatch.setenv("FINIC_LIFECYCLE_TESTS", "1")
    output = tmp_path / "evidence.json"
    output.write_text("previous attempt")
    with pytest.raises(FileExistsError):
        asyncio.run(runner()["run_live"](output))
    assert output.read_text() == "previous attempt"


@pytest.mark.parametrize("optimize", [0, 1])
@pytest.mark.parametrize(
    "failure", [None, "creation", "calibration", "listing", "identity", "matching", "score1", "score2"]
)
def test_lifecycle_sequence_and_evidence(failure, optimize, monkeypatch, tmp_path):
    module = runner(optimize)
    evidence = []
    calls = []
    ids = []

    def handle(request):
        calls.append((request.method, request.url.path))
        payload = json.loads(request.content) if request.content else None
        if request.url.path == "/v1/scores" and request.method == "POST":
            assert len(payload["data"]) == 100
            return httpx.Response(
                202, json={"model_id": "scm_new", "job_id": "job_creation", "request_id": "req_creation"}
            )
        if request.url.path.startswith("/v1/jobs/"):
            job_id = request.url.path.rsplit("/", 1)[-1]
            body = {"job_id": job_id, "complete": True}
            if job_id == "job_" + str(failure):
                body["error"] = {"type": "internal_error", "message": "provider detail"}
            return httpx.Response(200, json=body)
        if request.url.path == "/v1/scores":
            return httpx.Response(
                200,
                json={
                    "scores": [
                        {
                            "model_id": "scm_new",
                            "status": "training" if failure == "listing" else "ready",
                            "criteria": "synthetic",
                            "created_at": "2026-10-27",
                            "calibrated_at": None,
                            "examples": 100,
                            "bad_rate": 0.5,
                            "scored": 0,
                            "outcomes": 0,
                        }
                    ]
                },
            )
        if request.url.path == "/v1/score":
            assert payload["model_id"] == "scm_new"
            ids.append(payload["score_id"])
            if failure == "score" + str(len(ids)):
                persisted = json.loads(output.read_text())
                assert persisted["score_attempts"][-1] == {
                    "model_id": "scm_new",
                    "score_id": ids[-1],
                    "status": "pending",
                }
                raise httpx.ReadTimeout("private provider data", request=request)
            return httpx.Response(
                200,
                json={
                    "model_id": "scm_new",
                    "score_id": "wrong" if failure == "identity" else ids[-1],
                    "score": 50,
                    "request_id": "req_score",
                },
            )
        if request.url.path == "/v1/calibrate":
            assert payload["model_id"] == "scm_new"
            assert [item["score_id"] for item in payload["data"]] == ids
            return httpx.Response(
                202,
                json={
                    "job_id": "job_calibration",
                    "matched": 1 if failure == "matching" else 2,
                    "unmatched": 0,
                    "request_id": "req_calibration",
                },
            )
        raise AssertionError(request.url)

    output = tmp_path / "evidence.json"
    actual_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return actual_async_client(transport=httpx.MockTransport(handle), **kwargs)

    async def run():
        async with FinicClient("test", async_transport=httpx.MockTransport(handle)) as client:
            return await module["exercise_lifecycle"](client, lambda data: evidence.append(deepcopy(data)))

    if failure in ("score1", "score2"):
        monkeypatch.setenv("FINIC_API_KEY", "test-key")
        monkeypatch.setenv("FINIC_LIFECYCLE_TESTS", "1")
        monkeypatch.setattr(module["httpx"], "AsyncClient", client_factory)
        with pytest.raises(FinicError):
            asyncio.run(module["run_live"](output))
        persisted = json.loads(output.read_text())
        assert persisted["status"] == "failed"
        assert persisted["request_count"] == len(calls)
        assert len(ids) == int(failure[-1])
        assert len(persisted.get("scores", [])) == len(ids) - 1
        assert persisted["score_attempts"][-1]["status"] == "pending"
        assert all(item["status"] == "validated" for item in persisted["score_attempts"][:-1])
        assert ("POST", "/v1/calibrate") not in calls
        assert "test-key" not in output.read_text() and "private" not in output.read_text()
        return
    if failure in ("listing", "identity", "matching"):
        with pytest.raises(
            RuntimeError,
            match={"listing": "not listed as ready", "identity": "identity", "matching": "match both"}[
                failure
            ],
        ):
            asyncio.run(run())
        assert evidence[-1]["status"] != "passed"
        return
    if failure:
        with pytest.raises(FinicError, match="job failed"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result["status"] == "passed"
        assert len(ids) == len(set(ids)) == 2
        assert len(calls) == 7
        assert ("GET", "/v1/jobs/job_calibration") in calls
        assert [item["status"] for item in result["score_attempts"]] == ["validated", "validated"]
    assert evidence[0]["model_id"] == "scm_new"
    assert evidence[0]["creation_job_id"] == "job_creation"
    assert calls.count(("POST", "/v1/scores")) == 1
    if failure == "creation":
        assert not ids
        assert ("POST", "/v1/calibrate") not in calls
    else:
        assert evidence[-1]["calibration_job_id"] == "job_calibration"


def test_runner_records_ambiguous_failure_without_retry(monkeypatch, tmp_path):
    module = runner()
    monkeypatch.setenv("FINIC_API_KEY", "test-key")
    monkeypatch.setenv("FINIC_LIFECYCLE_TESTS", "1")
    actual_async_client = httpx.AsyncClient
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("private provider data", request=request)

    def client_factory(**kwargs):
        return actual_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(module["httpx"], "AsyncClient", client_factory)
    output = tmp_path / "evidence.json"
    with pytest.raises(FinicError):
        asyncio.run(module["run_live"](output))
    evidence = json.loads(output.read_text())
    assert evidence["status"] == "failed"
    assert evidence["request_count"] == len(calls) == 1
    assert "test-key" not in output.read_text() and "private" not in output.read_text()


def test_ci_does_not_select_lifecycle():
    makefile = (ROOT / "Makefile").read_text()
    ordinary = makefile.split("integration_test:\n")[1].split("\n", 1)[0]
    assert "-m live" in ordinary and "LIFECYCLE" not in ordinary
    workflow = (ROOT / ".github/workflows/integration.yml").read_text()
    assert "make integration_test" in workflow and "LIFECYCLE" not in workflow
