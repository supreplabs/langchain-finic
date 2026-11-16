"""Exercise disposable trial recovery without remote writes."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from langchain_finic import FinicError

example = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/disposable_scoring.py"))


class Client:
    def __init__(self, failure=None):
        self.failure = failure
        self.created = 0
        self.deleted = False
        self.scores = []

    async def acreate_scoring_model(self, request):
        self.created += 1
        return SimpleNamespace(model_id="owned", job_id="create-job")

    async def await_job(self, job_id, **kwargs):
        if self.failure == "job":
            raise RuntimeError("Terminal job failure")

    async def aget_job_status(self, job_id):
        return SimpleNamespace(complete=True)

    async def acalibrate(self, request):
        assert request.model_id == "owned"
        return SimpleNamespace(job_id="calibration-job")

    async def ascore(self, request):
        assert "is_bad" not in request.data
        self.scores.append(request.score_id)
        if self.failure == "score":
            raise TimeoutError("Ambiguous scoring write")
        return SimpleNamespace(score=600, questions=[])

    async def adelete_scoring_model(self, model_id):
        assert model_id == "owned"
        if self.failure == "delete":
            raise RuntimeError("Deletion failed")
        self.deleted = True

    async def aget_scoring_model(self, model_id):
        raise FinicError("Missing", status=404)

    async def alist_scoring_models(self):
        return SimpleNamespace(scoring_models=[SimpleNamespace(model_id="preexisting")])


async def exercise(client, path):
    rows = [{"amount": i, "is_bad": i % 2 == 0} for i in range(100)]
    return await example["run_trial"](
        client,
        training=rows,
        calibration=[{"amount": i + 100, "is_bad": i % 2 == 0} for i in range(100)],
        holdout=[{"amount": 300, "is_bad": True}, {"amount": 301, "is_bad": False}],
        criteria="Fictional trial",
        threshold=500,
        ledger_path=path,
    )


@pytest.mark.asyncio
async def test_complete_trial_and_exclusive_ledger(tmp_path):
    client = Client()
    path = tmp_path / "ledger.json"
    result = await exercise(client, path)
    assert result["status"] == "passed" and result["cleanup"] == "verified_deleted"
    assert client.deleted and len(set(client.scores)) == 4
    assert result["initial_trained"]["metrics"]["review_fraction"] == 1
    assert result["initial_trained"]["metrics"]["training_prior_brier"] == 0.25
    with pytest.raises(FileExistsError):
        await exercise(client, path)
    assert len(client.scores) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,deleted", [("job", True), ("score", False), ("delete", False)])
async def test_failure_and_cleanup_are_recorded_separately(tmp_path, failure, deleted):
    client = Client(failure)
    path = tmp_path / "ledger.json"
    with pytest.raises((RuntimeError, TimeoutError)):
        await exercise(client, path)
    ledger = json.loads(path.read_text())
    assert client.deleted is deleted
    assert ledger["cleanup"] == ("verified_deleted" if deleted else "manual_reconciliation_required")
    if failure == "score":
        assert ledger["pending_write"]["score_id"] == client.scores[0]
    if failure == "delete":
        assert ledger["status"] == "passed" and ledger["cleanup_error_type"] == "RuntimeError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [{"is_bad": True}, {"is_bad": False, "amount": float("nan")}, {"is_bad": True, "text": "x" * 60001}],
)
async def test_invalid_holdout_fails_before_any_remote_creation(tmp_path, invalid):
    client = Client()
    path = tmp_path / "ledger.json"
    rows = [{"amount": i, "is_bad": i % 2 == 0} for i in range(100)]
    with pytest.raises(ValueError):
        await example["run_trial"](
            client,
            training=rows,
            calibration=rows,
            holdout=[{"amount": 300, "is_bad": True}, invalid],
            criteria="Fictional trial",
            threshold=500,
            ledger_path=path,
        )
    assert client.created == 0
    assert client.scores == []
    assert not client.deleted
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "job"])
async def test_wait_observations_do_not_invent_progress(tmp_path, failure):
    path = tmp_path / "ledger.json"
    client = Client(failure)
    if failure:
        with pytest.raises(RuntimeError):
            await exercise(client, path)
    else:
        await exercise(client, path)
    waits = json.loads(path.read_text())["job_waits"]
    assert len(waits) == (1 if failure else 2)
    for observation in waits:
        assert observation["outcome"] == ("wait_raised" if failure else "completed")
        assert observation["elapsed_seconds"] >= 0
        assert not {"progress", "fraction", "stage", "eta"} & observation.keys()
    if failure:
        assert waits[0]["error_type"] == "RuntimeError"
