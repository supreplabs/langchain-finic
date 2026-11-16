"""Failure injection through the real HTTP client and disposable lifecycle runner."""

import asyncio
import json
import runpy
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from langchain_finic import FinicClient, FinicError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
run_trial = runpy.run_path(str(EXAMPLES / "disposable_scoring.py"))["run_trial"]
run_live = runpy.run_path(str(EXAMPLES / "public_dataset_scoring.py"))["run_live"]


def rows(offset=0):
    return [{"amount": offset + i, "is_bad": i % 2 == 0} for i in range(100)]


class Provider:
    """Protocol fixture; timeouts happen after recording server-side acceptance."""

    def __init__(self, failure=None):
        self.failure = failure
        self.calls = Counter()
        self.accepted = []
        self.poll_started = asyncio.Event()
        self.model = dict(
            model_id="owned",
            name="Disposable",
            status="ready",
            criteria="Fictional",
            created_at="2026-11-16",
            calibrated_at=None,
            archived_at=None,
            examples=100,
            bad_rate=0.5,
            scored=0,
        )

    async def __call__(self, request):
        method, path = request.method, request.url.path
        self.calls[(method, path)] += 1
        if method == "POST":
            body = json.loads(request.content)
            self.accepted.append((path, body))
            if self.failure == "create" and path == "/v1/scoring_models":
                raise httpx.ReadTimeout("private provider detail", request=request)
            if self.failure == "calibrate" and path == "/v1/calibrate":
                raise httpx.ReadTimeout("private provider detail", request=request)
            if path == "/v1/scoring_models":
                return httpx.Response(
                    200, json={"model_id": "owned", "job_id": "create-job", "request_id": "r"}
                )
            if path == "/v1/calibrate":
                return httpx.Response(200, json={"job_id": "calibration-job", "request_id": "r", "rows": 100})
            if path == "/v1/score":
                return httpx.Response(
                    200,
                    json={"model_id": "owned", "score_id": body["score_id"], "score": 600, "request_id": "r"},
                )
        if path.startswith("/v1/jobs/"):
            self.poll_started.set()
            body = {"job_id": path.rsplit("/", 1)[1], "complete": self.failure != "active"}
            if self.failure == "job_and_cleanup":
                body["error"] = {"type": "internal_error", "message": "private job detail"}
            return httpx.Response(200, json=body)
        if method == "DELETE":
            if self.failure in {"delete", "job_and_cleanup"}:
                raise httpx.ReadTimeout("private deletion detail", request=request)
            return httpx.Response(200, json={"model_id": "owned", "deleted": True})
        if path == "/v1/scoring_models/owned":
            return (
                httpx.Response(200, json=self.model) if self.failure == "get_exists" else httpx.Response(404)
            )
        if path == "/v1/scoring_models":
            return httpx.Response(
                200, json={"scoring_models": [self.model] if self.failure == "listed" else []}
            )
        raise AssertionError(f"Unexpected request: {method} {path}")


async def exercise(client, ledger):
    return await run_trial(
        client,
        training=rows(),
        calibration=rows(100),
        holdout=[{"amount": 300, "is_bad": True}],
        criteria="Fictional",
        threshold=500,
        ledger_path=ledger,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "calibrate"])
async def test_accepted_write_timeout_retains_evidence_and_never_replays(tmp_path, operation):
    provider = Provider(operation)
    path = tmp_path / "ledger.json"
    async with FinicClient("test", max_attempts=5, async_transport=httpx.MockTransport(provider)) as client:
        with pytest.raises(FinicError, match="outcome may be unknown"):
            await exercise(client, path)
        ledger = json.loads(path.read_text())
        assert ledger["pending_write"] == operation
        assert ledger["cleanup"] == "manual_reconciliation_required"
        assert ledger["model_name"].startswith("Disposable trial ")
        if operation == "calibrate":
            assert ledger["model_id"] == "owned" and ledger["creation_job_id"] == "create-job"
        else:
            assert "model_id" not in ledger  # The creation response was lost.
        calls = provider.calls.copy()
        with pytest.raises(FileExistsError):
            await exercise(client, path)
        assert provider.calls == calls
    endpoint = "/v1/scoring_models" if operation == "create" else "/v1/calibrate"
    assert provider.calls[("POST", endpoint)] == 1
    assert any(accepted_path == endpoint for accepted_path, _ in provider.accepted)
    assert provider.calls[("DELETE", "/v1/scoring_models/owned")] == 0
    assert "private" not in path.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["delete", "get_exists", "listed"])
async def test_cleanup_must_be_verified_not_just_attempted(tmp_path, failure):
    provider = Provider(failure)
    path = tmp_path / "ledger.json"
    async with FinicClient("test", max_attempts=5, async_transport=httpx.MockTransport(provider)) as client:
        with pytest.raises(RuntimeError, match="manual cleanup"):
            await exercise(client, path)
    ledger = json.loads(path.read_text())
    assert ledger["status"] == "passed"
    assert ledger["cleanup"] == "manual_reconciliation_required"
    assert "cleanup_error_type" in ledger
    assert provider.calls[("DELETE", "/v1/scoring_models/owned")] == 1
    if failure == "delete":
        assert provider.calls[("GET", "/v1/scoring_models/owned")] == 0
    if failure == "get_exists":
        assert provider.calls[("GET", "/v1/scoring_models")] == 0


@pytest.mark.asyncio
async def test_original_job_failure_survives_cleanup_timeout(tmp_path):
    provider = Provider("job_and_cleanup")
    path = tmp_path / "ledger.json"
    async with FinicClient("test", async_transport=httpx.MockTransport(provider)) as client:
        with pytest.raises(FinicError, match="job failed") as error:
            await exercise(client, path)
    assert error.value.error_type == "internal_error"
    ledger = json.loads(path.read_text())
    assert ledger["status"] == "failed" and ledger["cleanup"] == "manual_reconciliation_required"
    assert ledger["error_type"] == ledger["cleanup_error_type"] == "FinicError"
    assert "private" not in path.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_interrupted_wait_keeps_active_job_and_defers_deletion(tmp_path, interruption, monkeypatch):
    provider = Provider("active")
    path = tmp_path / "ledger.json"
    async with FinicClient("test", async_transport=httpx.MockTransport(provider)) as client:
        if interruption == "cancel":
            task = asyncio.create_task(exercise(client, path))
            await asyncio.wait_for(provider.poll_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            original = client.await_job

            async def bounded(job_id, **kwargs):
                return await original(job_id, timeout=0.01, poll_interval=0.001)

            monkeypatch.setattr(client, "await_job", bounded)
            with pytest.raises(TimeoutError):
                await exercise(client, path)
    ledger = json.loads(path.read_text())
    assert ledger["active_job"] == "create-job"
    assert ledger["job_waits"][0]["outcome"] == "wait_raised"
    assert ledger["error_type"] == ("CancelledError" if interruption == "cancel" else "TimeoutError")
    assert ledger["cleanup"] == "manual_reconciliation_required"
    assert provider.calls[("DELETE", "/v1/scoring_models/owned")] == 0
    assert not any(path == "/v1/score" for path, _ in provider.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [10, 11, 12])
async def test_budget_exhaustion_during_cleanup_remains_actionable(tmp_path, monkeypatch, budget):
    monkeypatch.setenv("FINIC_API_KEY", "test")
    provider = Provider()
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(provider), **kwargs)
    )
    monkeypatch.setitem(sys.modules, "disposable_scoring", SimpleNamespace(run_trial=run_trial))
    splits = {
        "train": [{"features": {"amount": i}, "is_bad": i % 2 == 0} for i in range(100)],
        "calibration": [{"features": {"amount": i + 100}, "is_bad": i % 2 == 0} for i in range(100)],
        "holdout": [{"features": {"amount": i + 200}, "is_bad": i % 2 == 0} for i in range(3)],
    }
    with pytest.raises(RuntimeError, match="manual cleanup"):
        await run_live("bank", splits, tmp_path, holdout_limit=3, threshold=500, request_budget=budget)
    ledger = json.loads((tmp_path / "lifecycle-ledger.json").read_text())
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert sum(provider.calls.values()) == budget == execution["request_attempts"]
    assert ledger["cleanup"] == "manual_reconciliation_required" and ledger["model_id"] == "owned"
    assert execution["status"] == "failed"
    assert provider.calls[("DELETE", "/v1/scoring_models/owned")] == (0 if budget == 10 else 1)
