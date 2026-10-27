import asyncio
import json
from concurrent.futures import CancelledError
from threading import Event

import httpx
import pytest
from pydantic import ValidationError

from langchain_finic import (
    CalibrationRequest,
    CreateScoreRequest,
    FinicClient,
    FinicError,
    ScoringRequest,
)


def examples():
    return [{"amount": i, "is_bad": i % 2 == 0} for i in range(100)]


def operation(name):
    return {
        "score": (
            ScoringRequest(model_id="scm_test", score_id="record", data={"amount": 10}),
            {"model_id": "scm_test", "score_id": "record", "score": 42, "request_id": "req_test"},
            "POST",
            "/v1/score",
        ),
        "create_score": (
            CreateScoreRequest(data=examples()),
            {"model_id": "scm_test", "job_id": "job_test", "request_id": "req_test"},
            "POST",
            "/v1/scores",
        ),
        "list_scores": (
            None,
            {
                "scores": [
                    {
                        "model_id": "scm_test",
                        "status": "ready",
                        "criteria": "Test",
                        "created_at": "2026-10-27T00:00:00Z",
                        "calibrated_at": None,
                        "examples": 100,
                        "bad_rate": 0.5,
                        "scored": 1,
                        "outcomes": 0,
                    }
                ]
            },
            "GET",
            "/v1/scores",
        ),
        "calibrate": (
            CalibrationRequest(model_id="scm_test", data=[{"score_id": "record", "is_bad": False}]),
            {"job_id": "job_test", "matched": 1, "unmatched": 0, "request_id": "req_test"},
            "POST",
            "/v1/calibrate",
        ),
        "get_job_status": ("job_test", {"job_id": "job_test", "complete": True}, "GET", "/v1/jobs/job_test"),
    }[name]


OPERATIONS = ["score", "create_score", "list_scores", "calibrate", "get_job_status"]


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_endpoint_contracts(name, asynchronous):
    argument, body, method, path = operation(name)
    seen = []

    def handle(request):
        seen.append(request)
        assert request.method == method and request.url.path == path
        assert request.headers["authorization"] == "Bearer test-key"
        if method == "POST":
            assert json.loads(request.content) == argument.model_dump(mode="json", exclude_none=True)
        else:
            assert not request.content
        return httpx.Response(202 if name in {"create_score", "calibrate"} else 200, json=body)

    transport = httpx.MockTransport(handle)
    client = FinicClient("test-key", transport=transport, async_transport=transport)
    args = [] if argument is None else [argument]
    if asynchronous:

        async def run():
            async with client:
                return await getattr(client, "a" + name)(*args)

        result = asyncio.run(run())
    else:
        with client:
            result = getattr(client, name)(*args)
    assert (
        result.model_dump(exclude_none=True) == {k: v for k, v in body.items() if v is not None}
        or name == "list_scores"
    )
    assert len(seen) == 1
    assert client._closed


@pytest.mark.parametrize("name", ["score", "create_score", "calibrate"])
@pytest.mark.parametrize("failure", ["timeout", 429, 503, 409])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_writes_never_replayed(name, failure, asynchronous):
    argument, _, _, _ = operation(name)
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private payload", request=request)
        return httpx.Response(
            failure,
            headers={"X-Request-Id": "req_fail"},
            json={"error": {"type": "duplicate_score_id", "message": "private payload"}},
        )

    transport = httpx.MockTransport(handle)
    client = FinicClient("secret", max_attempts=5, transport=transport, async_transport=transport)
    with pytest.raises(FinicError) as caught:
        if asynchronous:

            async def run():
                async with client:
                    await getattr(client, "a" + name)(argument)

            asyncio.run(run())
        else:
            with client:
                getattr(client, name)(argument)
    assert len(calls) == 1
    assert "private" not in str(caught.value) and "secret" not in str(caught.value)
    if failure != "timeout":
        assert caught.value.error_type == "duplicate_score_id"
        assert caught.value.request_id == "req_fail"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_reads_retry(asynchronous, monkeypatch):
    monkeypatch.setattr("langchain_finic.client._delay", lambda _: 0)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json={"scores": []})

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)
    if asynchronous:

        async def run():
            async with client:
                return await client.alist_scores()

        result = asyncio.run(run())
    else:
        with client:
            result = client.list_scores()
    assert result.scores == [] and len(calls) == 2


@pytest.mark.parametrize(
    "change", [{"score_id": "wrong"}, {"model_id": "wrong"}, {"score": 1001}, {"score": 0.1}, {"score": True}]
)
def test_reject_invalid_score_response(change):
    request, body, _, _ = operation("score")
    with FinicClient(
        "test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body | change))
    ) as client:
        with pytest.raises(FinicError):
            client.score(request)


@pytest.mark.parametrize("data", [{"amount": 1}, [{"amount": 1}], "amount\n1\n"])
def test_scoring_forms(data):
    request = ScoringRequest(
        model_id="m", score_id="s", data=data, skip_schema_validation=True, update_schema=False
    )
    assert request.data == data
    assert request.skip_schema_validation is True and request.update_schema is False


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{}, {}],
        {},
        {"is_bad": True},
        "x,x\n1,2",
        "x,y\n1",
        "x\n1\n2",
        {"a": "é" * 30000},
        {"a": float("inf")},
    ],
)
def test_invalid_scoring_data(data):
    with pytest.raises(ValidationError):
        ScoringRequest(model_id="m", score_id="s", data=data)


def test_training_json_csv_and_sampling():
    assert (
        len(CreateScoreRequest(data=examples() * 21).data) == 2100
    )  # Provider samples, client never truncates.
    csv = "amount,is_bad\n" + "".join(f"{i},{i % 2}\n" for i in range(100))
    assert CreateScoreRequest(data=csv).data == csv


@pytest.mark.parametrize(
    "data",
    [
        examples()[:99],
        [{"amount": i, "is_bad": False} for i in range(100)],
        [{"amount": i, "is_bad": "true"} for i in range(100)],
        [{"is_bad": i % 2} for i in range(100)],
        [{"amount": i} for i in range(100)],
    ],
)
def test_training_rejects_bad_examples(data):
    with pytest.raises(ValidationError):
        CreateScoreRequest(data=data)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"score_id": "s", "is_bad": "maybe"}],
        [{"score_id": "s", "is_bad": True, "model_id": "other"}],
        [{"score_id": "s" * 201, "is_bad": 0}],
    ],
)
def test_invalid_outcomes(data):
    with pytest.raises(ValidationError):
        CalibrationRequest(model_id="m", data=data)


def test_csv_outcomes():
    assert CalibrationRequest(model_id="m", data="score_id,is_bad\ns,0\n").data.endswith("s,0\n")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_wait_pending_then_complete(asynchronous, failed):
    calls = []

    def handle(request):
        calls.append(request)
        body = {"job_id": "job", "complete": len(calls) == 2}
        if len(calls) == 2 and failed:
            body["error"] = {"type": "unavailable", "message": "private content"}
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)

    def run():
        if asynchronous:

            async def arun():
                async with client:
                    return await client.await_job("job", timeout=1, poll_interval=0.001)

            return asyncio.run(arun())
        with client:
            return client.wait_for_job("job", timeout=1, poll_interval=0.001)

    if failed:
        with pytest.raises(FinicError) as exc:
            run()
        assert exc.value.error_type == "unavailable"
        assert "private" not in str(exc.value)
    else:
        assert run().succeeded
    assert len(calls) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
def test_wait_deadline(asynchronous):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"job_id": "job", "complete": False})

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)
    with pytest.raises(TimeoutError):
        if asynchronous:

            async def run():
                async with client:
                    await client.await_job("job", timeout=0.01, poll_interval=30)

            asyncio.run(run())
        else:
            with client:
                client.wait_for_job("job", timeout=0.01)
    assert len(calls) == 1


def test_sync_cancellation():
    event = Event()

    def handle(request):
        event.set()
        return httpx.Response(200, json={"job_id": "job", "complete": False})

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(CancelledError):
            client.wait_for_job("job", cancel_event=event)


def test_async_cancellation():
    async def run():
        started = asyncio.Event()
        calls = []

        async def handle(request):
            calls.append(request)
            started.set()
            return httpx.Response(200, json={"job_id": "job", "complete": False})

        async with FinicClient("test", async_transport=httpx.MockTransport(handle)) as client:
            task = asyncio.create_task(client.await_job("job"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(calls) == 1

    asyncio.run(run())


def test_body_limit_before_network():
    request = CreateScoreRequest(data=[{"value": "x" * 21000, "is_bad": i % 2} for i in range(100)])
    with FinicClient("test") as client:
        with pytest.raises(FinicError, match="2 MB"):
            client.create_score(request)
        assert client._sync_client is None


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_malformed_response_and_borrowed_clients(name, asynchronous):
    argument, _, _, _ = operation(name)
    args = [] if argument is None else [argument]
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, content=b"not JSON", headers={"X-Request-Id": "req_bad"})
    )
    if asynchronous:

        async def run():
            async with httpx.AsyncClient(transport=transport) as http:
                async with FinicClient("test", http_async_client=http) as client:
                    with pytest.raises(FinicError) as exc:
                        await getattr(client, "a" + name)(*args)
                    assert exc.value.request_id == "req_bad"
                assert not http.is_closed

        asyncio.run(run())
    else:
        with httpx.Client(transport=transport) as http:
            with FinicClient("test", http_client=http) as client:
                with pytest.raises(FinicError) as exc:
                    getattr(client, name)(*args)
                assert exc.value.request_id == "req_bad"
            assert not http.is_closed


@pytest.mark.parametrize(
    "name,code,error_type",
    [
        ("score", 400, "schema_mismatch"),
        ("score", 404, "score_not_found"),
        ("get_job_status", 404, "job_not_found"),
        ("calibrate", 409, "calibration_in_progress"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_provider_failures(name, code, error_type, asynchronous):
    argument, _, _, _ = operation(name)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            code, json={"error": {"type": error_type, "param": "data", "message": "private"}}
        )
    )
    client = FinicClient("test", transport=transport, async_transport=transport)
    with pytest.raises(FinicError) as exc:
        if asynchronous:

            async def run():
                async with client:
                    await getattr(client, "a" + name)(argument)

            asyncio.run(run())
        else:
            with client:
                getattr(client, name)(argument)
    assert exc.value.status == code and exc.value.error_type == error_type
    assert exc.value.param == "data" and "private" not in str(exc.value)
