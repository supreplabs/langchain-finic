"""Additional endpoint boundaries and request identity failures."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from langchain_finic import CreateScoreRequest, FinicClient, FinicError, JobStatus, ScoringRequest


@pytest.mark.parametrize("asynchronous", [False, True])
def test_job_identity_and_escaped_path(asynchronous):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.raw_path == b"/v1/jobs/job%2Fsegment%3Fq%3D1"
        return httpx.Response(
            200, headers={"X-Request-Id": "req_mismatch"}, json={"job_id": "different", "complete": True}
        )

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)
    with pytest.raises(FinicError, match="identity") as exc:
        if asynchronous:

            async def run():
                async with client:
                    await client.aget_job_status("job/segment?q=1")

            asyncio.run(run())
        else:
            with client:
                client.get_job_status("job/segment?q=1")
    assert len(calls) == 1 and exc.value.request_id == "req_mismatch"


@pytest.mark.parametrize(
    "job",
    [
        {"job_id": "j", "complete": "false"},
        {"job_id": "j", "complete": False, "error": {"type": "unavailable", "message": "Try later"}},
    ],
)
def test_malformed_job_status(job):
    with pytest.raises(ValidationError):
        JobStatus.model_validate(job)


@pytest.mark.parametrize("timeout,interval", [(0, 30), (1, 0), (float("inf"), 30), (1, float("nan"))])
def test_invalid_poll_options_do_not_send(timeout, interval):
    with FinicClient("test") as client:
        with pytest.raises(ValueError):
            client.wait_for_job("j", timeout=timeout, poll_interval=interval)
        assert client._sync_client is None


def test_json_size_boundary_and_ignored_label():
    overhead = len(json.dumps({"x": ""}, separators=(",", ":")).encode())
    data = {"x": "a" * (60_000 - overhead), "is_bad": "ignored-by-scoring"}
    assert ScoringRequest(model_id="m", score_id="s", data=data).data == data
    data["x"] += "a"
    with pytest.raises(ValidationError):
        ScoringRequest(model_id="m", score_id="s", data=data)


@pytest.mark.parametrize(
    "field,value",
    [("score_id", "x" * 201), ("score_id", " "), ("skip_schema_validation", "true"), ("update_schema", 1)],
)
def test_scoring_field_boundaries(field, value):
    with pytest.raises(ValidationError):
        ScoringRequest.model_validate({"model_id": "m", "score_id": "s", "data": {"x": 1}, field: value})


def test_description_boundary():
    rows = [{"x": i, "is_bad": i % 2} for i in range(100)]
    assert len(CreateScoreRequest(data=rows, description="x" * 4000).description) == 4000
    with pytest.raises(ValidationError):
        CreateScoreRequest(data=rows, description="x" * 4001)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_poll_transport_failure_does_not_imply_unknown_write(asynchronous):
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("connection lost", request=request)

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)
    with pytest.raises(FinicError) as exc:
        if asynchronous:

            async def run():
                async with client:
                    await client.await_job("job")

            asyncio.run(run())
        else:
            with client:
                client.wait_for_job("job")
    assert len(calls) == 1
    assert "write" not in str(exc.value)
