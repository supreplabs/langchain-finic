import asyncio
import json
from concurrent.futures import CancelledError
from threading import Event

import httpx
import pytest
from pydantic import ValidationError

from langchain_finic import (
    AddQuestionsRequest,
    CalibrationRequest,
    CreateScoringModelRequest,
    FinicClient,
    FinicError,
    ScoringRequest,
    UpdateScoringModelRequest,
)


def examples():
    return [{"amount": i, "is_bad": i % 2 == 0} for i in range(100)]


def summary(**overrides):
    return {
        "model_id": "scm_test",
        "name": "Card-not-present payment fraud",
        "status": "ready",
        "criteria": "Test",
        "created_at": "2026-10-27T00:00:00Z",
        "calibrated_at": None,
        "archived_at": None,
        "examples": 100,
        "bad_rate": 0.5,
        "scored": 1,
    } | overrides


def detail(**overrides):
    questions = [{"id": "q001", "text": "Is it new?", "source": "user", "weight": 0.93}]
    return summary(questions=questions) | overrides


def without_none(value):
    """Mirror model_dump(exclude_none=True), which drops nested nulls too."""
    if isinstance(value, dict):
        return {key: without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_none(item) for item in value]
    return value


def operation(name):
    return {
        "score": (
            (ScoringRequest(model_id="scm_test", score_id="record", data={"amount": 10}),),
            {"model_id": "scm_test", "score_id": "record", "score": 42, "request_id": "req_test"},
            "POST",
            "/v1/score",
            200,
        ),
        "create_scoring_model": (
            (CreateScoringModelRequest(name="Fraud", data=examples()),),
            {"model_id": "scm_test", "job_id": "job_test", "request_id": "req_test"},
            "POST",
            "/v1/scoring_models",
            202,
        ),
        "list_scoring_models": (
            (),
            {"scoring_models": [summary()]},
            "GET",
            "/v1/scoring_models",
            200,
        ),
        "get_scoring_model": (("scm_test",), detail(), "GET", "/v1/scoring_models/scm_test", 200),
        "update_scoring_model": (
            ("scm_test", UpdateScoringModelRequest(archived=True)),
            detail(archived_at="2026-11-13T00:00:00Z"),
            "PATCH",
            "/v1/scoring_models/scm_test",
            200,
        ),
        "delete_scoring_model": (
            ("scm_test",),
            {"model_id": "scm_test", "deleted": True},
            "DELETE",
            "/v1/scoring_models/scm_test",
            200,
        ),
        "add_questions": (
            ("scm_test", AddQuestionsRequest(questions=["Is it new?"])),
            detail(added=[{"id": "q001", "text": "Is it new?", "used": True}]),
            "POST",
            "/v1/scoring_models/scm_test/questions",
            200,
        ),
        "remove_question": (
            ("scm_test", "q001"),
            detail(),
            "DELETE",
            "/v1/scoring_models/scm_test/questions/q001",
            200,
        ),
        "calibrate": (
            (CalibrationRequest(model_id="scm_test", data=examples()),),
            {"job_id": "job_test", "rows": 100, "request_id": "req_test"},
            "POST",
            "/v1/calibrate",
            202,
        ),
        "get_job_status": (
            ("job_test",),
            {"job_id": "job_test", "complete": True},
            "GET",
            "/v1/jobs/job_test",
            200,
        ),
    }[name]


OPERATIONS = [
    "score",
    "create_scoring_model",
    "list_scoring_models",
    "get_scoring_model",
    "update_scoring_model",
    "delete_scoring_model",
    "add_questions",
    "remove_question",
    "calibrate",
    "get_job_status",
]
# Calls that carry a request body, so the serialized request can be asserted.
WITH_BODY = {"score", "create_scoring_model", "update_scoring_model", "add_questions", "calibrate"}
# Writes whose repetition the provider rejects or whose outcome is not knowable.
WRITES = ["score", "create_scoring_model", "delete_scoring_model", "add_questions", "calibrate"]


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_endpoint_contracts(name, asynchronous):
    args, body, method, path, status = operation(name)
    seen = []

    def handle(request):
        seen.append(request)
        assert request.method == method and request.url.path == path
        assert request.headers["authorization"] == "Bearer test-key"
        if name in WITH_BODY:
            assert json.loads(request.content) == args[-1].model_dump(mode="json", exclude_none=True)
        else:
            assert not request.content
        return httpx.Response(status, json=body)

    transport = httpx.MockTransport(handle)
    client = FinicClient("test-key", transport=transport, async_transport=transport)
    if asynchronous:

        async def run():
            async with client:
                return await getattr(client, "a" + name)(*args)

        result = asyncio.run(run())
    else:
        with client:
            result = getattr(client, name)(*args)
    assert result.model_dump(mode="json", exclude_none=True) == without_none(body)
    assert len(seen) == 1
    assert client._closed


def test_criteria_only_creation_has_no_job():
    body = {"model_id": "scm_test", "request_id": "req_test"}
    request = CreateScoringModelRequest(name="Fraud", criteria="Chargebacks within 90 days.")
    with FinicClient(
        "test", transport=httpx.MockTransport(lambda _: httpx.Response(201, json=body))
    ) as client:
        result = client.create_scoring_model(request)
    assert result.job_id is None and result.ready


@pytest.mark.parametrize("name", ["get_scoring_model", "update_scoring_model", "remove_question"])
def test_scoring_model_identity_is_checked(name):
    args, body, _, _, _ = operation(name)
    wrong = body | {"model_id": "scm_other"}
    with FinicClient(
        "test",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"X-Request-Id": "req_x"}, json=wrong)
        ),
    ) as client:
        with pytest.raises(FinicError, match="identity") as exc:
            getattr(client, name)(*args)
    assert exc.value.request_id == "req_x"


@pytest.mark.parametrize("method,args", [("get_scoring_model", (" ",)), ("remove_question", ("m", ""))])
def test_blank_path_segments_do_not_send(method, args):
    with FinicClient("test") as client:
        with pytest.raises(ValueError):
            getattr(client, method)(*args)
        assert client._sync_client is None


def test_path_segments_are_escaped():
    seen = []

    def handle(request):
        seen.append(request.url.raw_path)
        return httpx.Response(200, json=detail(model_id="a/b"))

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        client.remove_question("a/b", "q 1?")
    assert seen == [b"/v1/scoring_models/a%2Fb/questions/q%201%3F"]


@pytest.mark.parametrize("name", WRITES)
@pytest.mark.parametrize("failure", ["timeout", 429, 503, 409])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_writes_never_replayed(name, failure, asynchronous):
    args, _, _, _, _ = operation(name)
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
                    await getattr(client, "a" + name)(*args)

            asyncio.run(run())
        else:
            with client:
                getattr(client, name)(*args)
    assert len(calls) == 1
    assert "private" not in str(caught.value) and "secret" not in str(caught.value)
    if failure == "timeout":
        assert "write outcome may be unknown" in str(caught.value)
    else:
        assert caught.value.error_type == "duplicate_score_id"
        assert caught.value.request_id == "req_fail"


@pytest.mark.parametrize(
    "name,attempts", [("list_scoring_models", 2), ("get_scoring_model", 2), ("update_scoring_model", 2)]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_reads_and_idempotent_updates_retry(name, attempts, asynchronous, monkeypatch):
    monkeypatch.setattr("langchain_finic.client._delay", lambda _: 0)
    args, body, _, _, _ = operation(name)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=body)

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)
    if asynchronous:

        async def run():
            async with client:
                return await getattr(client, "a" + name)(*args)

        asyncio.run(run())
    else:
        with client:
            getattr(client, name)(*args)
    assert len(calls) == attempts


@pytest.mark.parametrize(
    "change", [{"score_id": "wrong"}, {"model_id": "wrong"}, {"score": 1001}, {"score": 0.1}, {"score": True}]
)
def test_reject_invalid_score_response(change):
    (request,), body, _, _, _ = operation("score")
    with FinicClient(
        "test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body | change))
    ) as client:
        with pytest.raises(FinicError):
            client.score(request)


def test_include_questions_round_trip():
    request = ScoringRequest(
        model_id="scm_test", score_id="record", data={"amount": 10}, include_questions=True
    )
    body = {
        "model_id": "scm_test",
        "score_id": "record",
        "score": 874,
        "questions": [{"id": "q004", "text": "Different country?", "probability": 0.8, "weight": 0.91}],
        "request_id": "req_test",
    }
    seen = []

    def handle(request_):
        seen.append(json.loads(request_.content))
        return httpx.Response(200, json=body)

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        result = client.score(request)
    assert seen[0]["include_questions"] is True
    assert result.questions is not None
    assert result.questions[0].id == "q004" and result.questions[0].weight == 0.91


def test_unrequested_questions_are_rejected():
    (request,), body, _, _, _ = operation("score")
    unwanted = body | {
        "questions": [{"id": "q004", "text": "Different country?", "probability": 0.8, "weight": 0.9}]
    }
    with FinicClient(
        "test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=unwanted))
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


def test_labeled_json_csv_and_sampling():
    assert (
        len(CreateScoringModelRequest(name="M", data=examples() * 21).data) == 2100
    )  # Provider samples, client never truncates.
    csv = "amount,is_bad\n" + "".join(f"{i},{i % 2}\n" for i in range(100))
    assert CreateScoringModelRequest(name="M", data=csv).data == csv


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
def test_labeled_records_reject_bad_examples(data):
    with pytest.raises(ValidationError):
        CreateScoringModelRequest(name="M", data=data)
    with pytest.raises(ValidationError):
        CalibrationRequest(model_id="m", data=data)


def test_creation_needs_criteria_or_records():
    with pytest.raises(ValidationError):
        CreateScoringModelRequest(name="M")
    with pytest.raises(ValidationError):
        CreateScoringModelRequest(name="M", criteria="   ")
    assert CreateScoringModelRequest(name="M", criteria="Chargebacks.").data is None


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"name": " "},
        {"name": "x" * 101},
        {"criteria": "x" * 4001},
        {"archived": "yes"},
    ],
)
def test_invalid_updates(arguments):
    with pytest.raises(ValidationError):
        UpdateScoringModelRequest.model_validate(arguments)


def test_unarchive_survives_exclude_none():
    request = UpdateScoringModelRequest(archived=False)
    assert request.model_dump_json(exclude_none=True) == '{"archived":false}'


@pytest.mark.parametrize("questions", [[], ["  "], ["x" * 501], ["q"] * 21])
def test_invalid_added_questions(questions):
    with pytest.raises(ValidationError):
        AddQuestionsRequest(questions=questions)


def test_calibration_question_changes():
    request = CalibrationRequest(
        model_id="m", data=examples(), add_questions=["New?"], remove_questions=["q1"]
    )
    assert request.add_questions == ["New?"] and request.remove_questions == ["q1"]
    with pytest.raises(ValidationError):
        CalibrationRequest(model_id="m", data=examples(), remove_questions=["q1", "q1"])


def test_csv_labeled_records():
    csv = "amount,is_bad\n" + "".join(f"{i},{i % 2}\n" for i in range(100))
    assert CalibrationRequest(model_id="m", data=csv).data == csv


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
    request = CreateScoringModelRequest(
        name="M", data=[{"value": "x" * 21000, "is_bad": i % 2} for i in range(100)]
    )
    with FinicClient("test") as client:
        with pytest.raises(FinicError, match="2 MB"):
            client.create_scoring_model(request)
        assert client._sync_client is None


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_malformed_response_and_borrowed_clients(name, asynchronous):
    args, _, _, _, _ = operation(name)
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
        ("score", 404, "scoring_model_not_found"),
        ("score", 409, "no_questions"),
        ("get_job_status", 404, "job_not_found"),
        ("calibrate", 409, "calibration_in_progress"),
        ("delete_scoring_model", 403, "permission_denied"),
        ("remove_question", 409, "question_in_use"),
        ("add_questions", 409, "conflict"),
        ("update_scoring_model", 404, "scoring_model_not_found"),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_provider_failures(name, code, error_type, asynchronous):
    args, _, _, _, _ = operation(name)
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
                    await getattr(client, "a" + name)(*args)

            asyncio.run(run())
        else:
            with client:
                getattr(client, name)(*args)
    assert exc.value.status == code and exc.value.error_type == error_type
    assert exc.value.param == "data" and "private" not in str(exc.value)
