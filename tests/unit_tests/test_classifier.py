import asyncio
import json

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import ValidationError

from langchain_finic import (
    Choice,
    DecisionRequest,
    DecisionResponse,
    FinicClassifier,
    FinicClient,
    FinicError,
    MultiChoice,
    Noul,
    Score,
)


@pytest.mark.parametrize("key", [None, "configured-key", ""])
def test_api_key_environment(monkeypatch, key):
    monkeypatch.delenv("FINIC_API_KEY", raising=False)
    monkeypatch.setenv("GRAYSON_API_KEY", "ignored")
    if key is not None:
        monkeypatch.setenv("FINIC_API_KEY", key)
    if not key:
        with pytest.raises(ValidationError):
            FinicClient()
    else:
        with FinicClient() as client:
            assert client.config.api_key.get_secret_value() == key
    with FinicClient("explicit-key") as client:
        assert client.config.api_key.get_secret_value() == "explicit-key"


def test_sync_contract_and_credentials(request_body, response_body):
    def handle(request):
        assert str(request.url) == "https://api.finic.ai/v1/decide"
        assert request.headers["authorization"] == "Bearer fake-key"
        assert "base_rate" not in json.loads(request.content)["questions"]["fits_profile"]
        return httpx.Response(200, json=response_body)

    model = FinicClassifier(api_key="fake-key", transport=httpx.MockTransport(handle))
    result = model.invoke(request_body.model_dump())
    assert isinstance(result, DecisionResponse)
    assert result.answers["action"].choice == "escalate"
    assert "fake-key" not in repr(model)
    assert "fake-key" not in repr(model._client.config)


def test_lcel_callbacks_batch_and_tool(request_body, handler):
    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.tags = []

        def on_chain_start(self, serialized, inputs, **kwargs):
            self.tags.extend(kwargs.get("tags", []))

    callback = Recorder()
    classifier = FinicClassifier(api_key="fake-key", transport=httpx.MockTransport(handler))
    chain = classifier | (lambda result: result.answers["action"].choice)
    assert chain.invoke(request_body, {"callbacks": [callback], "tags": ["aml"]}) == "escalate"
    assert "aml" in callback.tags
    assert len(classifier.batch([request_body, request_body], {"max_concurrency": 1})) == 2
    tool = classifier.as_tool()
    assert tool.invoke(request_body.model_dump())["answers"]["action"]["choice"] == "escalate"
    message = tool.invoke(
        {
            "name": tool.name,
            "args": request_body.model_dump(mode="json"),
            "id": "call_1",
            "type": "tool_call",
        }
    )
    assert json.loads(message.content)["request_id"] == "req_test"
    assert classifier.get_input_schema().model_json_schema()["title"] == "DecisionRequest"


def test_native_async_batch_tool_and_stream(request_body, handler):
    async def async_handler(request):
        await asyncio.sleep(0)
        return handler(request)

    classifier = FinicClassifier(api_key="fake-key", async_transport=httpx.MockTransport(async_handler))

    async def run():
        assert (await classifier.ainvoke(request_body)).id == "dec_test"
        assert len(await classifier.abatch([request_body] * 2, {"max_concurrency": 1})) == 2
        assert (await classifier.as_tool().ainvoke(request_body.model_dump()))["id"] == "dec_test"
        assert len([result async for result in classifier.astream(request_body)]) == 1

    asyncio.run(run())


@pytest.mark.parametrize("status", [400, 401, 402, 404, 413])
def test_permanent_errors_do_not_retry(status, request_body):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "sensitive-case-data"}})

    client = FinicClient("fake", transport=httpx.MockTransport(handle))
    with pytest.raises(FinicError) as exc:
        client.decide(request_body)
    assert len(calls) == 1
    assert exc.value.status == status
    assert "sensitive-case-data" not in str(exc.value)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_errors_retry(status, request_body, response_body, monkeypatch):
    monkeypatch.setattr("langchain_finic.client.time.sleep", lambda _: None)
    responses = [httpx.Response(status, text="temporary"), httpx.Response(200, json=response_body)]
    client = FinicClient("fake", transport=httpx.MockTransport(lambda _: responses.pop(0)))
    assert client.decide(request_body).id == "dec_test"
    assert not responses


def test_network_failure_is_bounded(request_body, monkeypatch):
    monkeypatch.setattr("langchain_finic.client.time.sleep", lambda _: None)
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ConnectError("private-network-detail")

    client = FinicClient("fake", max_attempts=2, transport=httpx.MockTransport(handle))
    with pytest.raises(FinicError, match="bounded retries"):
        client.decide(request_body)
    assert len(calls) == 2


def test_malformed_response_fails_closed(request_body, response_body):
    response_body["answers"].pop("action")
    client = FinicClient(
        "fake", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response_body))
    )
    with pytest.raises(FinicError, match="contract"):
        client.decide(request_body)


@pytest.mark.parametrize("state", ["Evidence", {"events": []}, ["event"]])
def test_generic_input_types(state):
    request = DecisionRequest(state=state, questions={"risk": Noul(instructions="Is there a risk?")})
    assert request.state == state


def test_question_validation():
    with pytest.raises(ValidationError):
        Choice(instructions="Pick", criteria={"a": " Same ", "b": "same"})
    with pytest.raises(ValidationError):
        Choice(instructions="Pick", criteria={"a": "A", "b": "B"}, base_rate={"a": 0.9, "b": 0.9})
    with pytest.raises(ValidationError):
        Score(instructions="Rate", criteria=["low", "high"], base_rate=[1])
    with pytest.raises(ValidationError):
        DecisionRequest(state=" ", questions={"risk": Noul(instructions="Risk?")})
    with pytest.raises(ValidationError):
        DecisionRequest(state="Case", questions={"1bad": Noul(instructions="Risk?")})
    assert MultiChoice(instructions="Choose", criteria={"a": "A", "b": "B"}, base_rate={"a": 0.8})


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_bad_probabilities_rejected(response_body, bad):
    response_body["answers"]["fits_profile"]["noul"] = bad
    with pytest.raises(ValidationError):
        DecisionResponse.model_validate(response_body)


def test_inconsistent_distribution_rejected(response_body):
    response_body["answers"]["action"]["probabilities"]["close"] = 0.95
    with pytest.raises(ValidationError):
        DecisionResponse.model_validate(response_body)


def test_async_retry_and_error(request_body, response_body, monkeypatch):
    async def no_delay(_):
        pass

    monkeypatch.setattr("langchain_finic.client.asyncio.sleep", no_delay)
    responses = [httpx.Response(503), httpx.Response(200, json=response_body)]
    client = FinicClient("fake", async_transport=httpx.MockTransport(lambda _: responses.pop(0)))
    assert asyncio.run(client.adecide(request_body)).id == "dec_test"
    bad = FinicClient("fake", async_transport=httpx.MockTransport(lambda _: httpx.Response(401)))
    with pytest.raises(FinicError):
        asyncio.run(bad.adecide(request_body))
