"""Contract coverage for the current Finic scoring-model API."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from langchain_finic import (
    AddQuestionsRequest,
    CalibrationRequest,
    CreateScoringModelRequest,
    DecisionRequest,
    FinicClient,
    FinicError,
    FinicToolkit,
    ScoringRequest,
    UpdateScoringModelRequest,
)


def labeled_data():
    return [{"amount": index, "is_bad": index % 2 == 0} for index in range(100)]


def scoring_model(**changes):
    return {
        "model_id": "scm_test",
        "name": "Payment risk",
        "status": "ready",
        "criteria": "Bad means confirmed fraud",
        "created_at": "2026-11-01T00:00:00Z",
        "calibrated_at": None,
        "archived_at": None,
        "examples": 0,
        "bad_rate": 0,
        "scored": 0,
        "questions": [],
        **changes,
    }


@pytest.mark.parametrize("data", [None, "", [], [{"is_bad": True}]])
def test_create_model_requires_valid_creation_mode(data):
    values = {"name": "Payment risk"}
    if data is not None:
        values["data"] = data
    with pytest.raises(ValidationError):
        CreateScoringModelRequest.model_validate(values)


def test_create_model_supports_criteria_and_labeled_data():
    criteria = CreateScoringModelRequest(name="Payments", criteria="Bad means chargeback")
    assert criteria.criteria == "Bad means chargeback" and criteria.data is None
    trained = CreateScoringModelRequest(name="Payments", data=labeled_data())
    assert len(trained.data) == 100
    with pytest.raises(ValidationError):
        CreateScoringModelRequest(name="Payments", data=[{"amount": i, "is_bad": False} for i in range(100)])
    with pytest.raises(ValidationError, match="true or false"):
        CreateScoringModelRequest(
            name="Payments",
            data=[{"amount": i, "is_bad": "yes" if i % 2 else "no"} for i in range(100)],
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_scoring_model_endpoints_are_typed_and_path_ids_escaped(asynchronous):
    calls = []
    model = scoring_model(model_id="scm/segment", added=[{"id": "q1", "text": "New device?", "used": True}])
    paths = {
        "GET /v1/scoring_models": {"scoring_models": [scoring_model()]},
        "GET /v1/scoring_models/scm%2Fsegment": scoring_model(model_id="scm/segment"),
        "PATCH /v1/scoring_models/scm%2Fsegment": scoring_model(model_id="scm/segment"),
        "POST /v1/scoring_models/scm%2Fsegment/questions": model,
        "DELETE /v1/scoring_models/scm%2Fsegment/questions/q%2F1": scoring_model(model_id="scm/segment"),
        "DELETE /v1/scoring_models/scm%2Fsegment": {"model_id": "scm/segment", "deleted": True},
        "POST /v1/scoring_models": {
            "model_id": "scm_test",
            "job_id": "job_test",
            "request_id": "req_test",
        },
        "POST /v1/ask": {
            "id": "dec_test",
            "model": "grayson-1",
            "answers": {"risk": {"type": "noul", "noul": 0.25}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "request_id": "req_test",
            "evaluation_time_ms": 1,
        },
    }

    def handle(request):
        key = f"{request.method} {request.url.raw_path.decode().split(b'?'.decode())[0]}"
        calls.append(request)
        assert key in paths
        return httpx.Response(200, json=paths[key])

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)

    async def run():
        await client.alist_scoring_models()
        assert (await client.aget_scoring_model("scm/segment")).model_id == "scm/segment"
        assert (
            await client.aupdate_scoring_model("scm/segment", UpdateScoringModelRequest(archived=False))
        ).archived_at is None
        assert (
            (await client.aadd_questions("scm/segment", AddQuestionsRequest(questions=["New device?"])))
            .added[0]
            .used
        )
        await client.aremove_question("scm/segment", "q/1")
        assert (await client.adelete_scoring_model("scm/segment")).deleted
        assert (
            await client.acreate_scoring_model(
                CreateScoringModelRequest(name="Payments", data=labeled_data())
            )
        ).job_id == "job_test"
        ask = DecisionRequest(
            data={"amount": 1}, questions={"risk": {"type": "noul", "instructions": "Fraud?"}}
        )
        assert (await client.aask(ask)).request_id == "req_test"

    def sync_run():
        client.list_scoring_models()
        client.get_scoring_model("scm/segment")
        client.update_scoring_model("scm/segment", UpdateScoringModelRequest(archived=False))
        client.add_questions("scm/segment", AddQuestionsRequest(questions=["New device?"]))
        client.remove_question("scm/segment", "q/1")
        assert client.delete_scoring_model("scm/segment").deleted
        client.create_scoring_model(CreateScoringModelRequest(name="Payments", data=labeled_data()))
        ask = DecisionRequest(
            data={"amount": 1}, questions={"risk": {"type": "noul", "instructions": "Fraud?"}}
        )
        assert client.ask(ask).request_id == "req_test"

    if asynchronous:
        asyncio.run(run())
    else:
        sync_run()
    assert len(calls) == 8


def test_score_includes_optional_question_evidence_and_keeps_identity():
    request = ScoringRequest(
        model_id="scm_test", score_id="record_1", data={"amount": 1}, include_questions=True
    )
    body = {
        "model_id": "scm_test",
        "score_id": "record_1",
        "score": 250,
        "request_id": "req_test",
        "questions": [{"id": "q1", "text": "New device?", "probability": 0.4, "weight": 0.2}],
    }

    def handle(req):
        assert json.loads(req.content)["include_questions"] is True
        return httpx.Response(200, json=body)

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        assert client.score(request).questions[0].probability == 0.4


def test_calibration_current_rows_and_question_changes():
    request = CalibrationRequest(
        model_id="scm_test",
        data=labeled_data(),
        add_questions=["Is the account new?"],
        remove_questions=["q001"],
    )
    assert len(request.data) == 100 and request.remove_questions == ["q001"]
    earlier_name = CalibrationRequest(model_id="scm_test", examples=labeled_data())
    assert len(earlier_name.data) == 100


def test_calibration_sends_current_question_controls():
    request = CalibrationRequest(
        model_id="scm_test",
        data=labeled_data(),
        add_questions=["New device?"],
        remove_questions=["q001"],
    )
    body = {
        "job_id": "job_calibrate",
        "rows": 100,
        "request_id": "req_calibrate",
    }

    def handle(req):
        payload = json.loads(req.content)
        assert payload["add_questions"] == ["New device?"]
        assert payload["remove_questions"] == ["q001"]
        assert payload["model_id"] == "scm_test" and len(payload["data"]) == 100
        return httpx.Response(202, json=body)

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        assert client.calibrate(request).rows == 100


def test_criteria_only_create_returns_without_a_job():
    response = {"model_id": "scm_test", "request_id": "req_create"}

    with FinicClient(
        "test", transport=httpx.MockTransport(lambda _: httpx.Response(201, json=response))
    ) as client:
        result = client.create_scoring_model(
            CreateScoringModelRequest(name="Payments", criteria="Bad means confirmed fraud")
        )
    assert result.job_id is None


@pytest.mark.parametrize("payload", [{}, {"name": None}, {"archived": "false"}])
def test_update_model_requires_real_strict_change(payload):
    with pytest.raises(ValidationError):
        UpdateScoringModelRequest.model_validate(payload)


def test_write_endpoints_are_never_replayed_and_model_delete_is_explicit():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    with FinicClient("test", max_attempts=4, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FinicError):
            client.delete_scoring_model("scm_test")
    assert len(calls) == 1


def test_toolkit_adds_lifecycle_tools_only_when_selected():
    seen = []

    def handle(request):
        seen.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(200, json={"model_id": "scm_test", "deleted": True})
        return httpx.Response(200, json=scoring_model())

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        default_names = {tool.name for tool in FinicToolkit(client=client).get_tools()}
        assert default_names == {"finic_list_scoring_models", "finic_get_job_status"}
        selected = FinicToolkit(
            client=client, operations=("add_questions", "update_scoring_model", "delete_scoring_model")
        ).get_tools()
        tools = {tool.name: tool for tool in selected}
        tools["finic_add_questions"].invoke({"model_id": "scm_test", "questions": ["New device?"]})
        tools["finic_update_scoring_model"].invoke({"model_id": "scm_test", "archived": False})
        assert tools["finic_delete_scoring_model"].invoke({"model_id": "scm_test"})["deleted"]
    assert seen == [
        ("POST", "/v1/scoring_models/scm_test/questions"),
        ("PATCH", "/v1/scoring_models/scm_test"),
        ("DELETE", "/v1/scoring_models/scm_test"),
    ]
