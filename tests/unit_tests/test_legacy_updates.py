"""Regression cases discovered by the independent compatibility review."""

import asyncio
import inspect
import json

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumps, loads
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from langchain_finic import DecisionRequest, FinicClassifier, Noul


def request():
    return DecisionRequest(state={"amount": 10}, questions={"risk": Noul(instructions="Risk?")})


def response(http_request):
    body = json.loads(http_request.content)
    return httpx.Response(
        200,
        json={
            "id": "dec_legacy",
            "model": body["model"],
            "answers": {"risk": {"type": "noul", "noul": 0.2}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "request_id": "req_legacy",
            "evaluation_time_ms": 1,
        },
    )


@pytest.mark.parametrize("deep", [False, True])
def test_legacy_copy_sends_updated_evidence(deep):
    original = request()
    copied = original.model_copy(update={"state": {"amount": 10000}}, deep=deep)
    sent = []

    def handle(http_request):
        sent.append(json.loads(http_request.content))
        return response(http_request)

    with FinicClassifier(api_key="test", transport=httpx.MockTransport(handle)) as classifier:
        classifier.invoke(copied)
    assert sent[0]["data"] == {"amount": 10000}
    assert "state" not in sent[0]
    assert copied.state == copied.data == {"amount": 10000}
    assert original.state == {"amount": 10}
    assert "model" not in copied.model_fields_set


def test_legacy_assignment_normalizes_messages_and_tracks_fields():
    value = request()
    value.state = [HumanMessage(content="new evidence")]
    assert value.data == [{"role": "user", "content": "new evidence"}]
    assert "data" in value.model_fields_set and "state" not in value.model_fields_set
    assert "state" not in value.model_dump()
    previous = value.model_dump()
    with pytest.raises(ValidationError):
        value.state = None
    assert value.model_dump() == previous


@pytest.mark.parametrize(
    "update", [{"state": None}, {"state": 42}, {"state": " "}, {"state": "a", "data": "b"}]
)
def test_invalid_legacy_copy_is_rejected(update):
    original = request()
    with pytest.raises(ValueError):
        original.model_copy(update=update)
    assert original.data == {"amount": 10}


@pytest.mark.parametrize("conversion", ["direct", "schema", "configured", "bound", "combined"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_legacy_tool_conversion_paths(conversion, asynchronous):
    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.tags = []

        def on_chain_start(self, serialized, inputs, **kwargs):
            self.tags.extend(kwargs.get("tags", []))

    callback = Recorder()
    sent = []

    def handle(http_request):
        sent.append(json.loads(http_request.content))
        return response(http_request)

    transport = httpx.MockTransport(handle)
    classifier = FinicClassifier(api_key="test", transport=transport, async_transport=transport)
    runnable = classifier
    if conversion in {"configured", "combined"}:
        runnable = runnable.with_config(tags=["bound-tag"])
    if conversion in {"bound", "combined"}:
        runnable = runnable.bind(model="grayson-test")
    tool = runnable.as_tool(args_schema=DecisionRequest) if conversion == "schema" else runnable.as_tool()
    payload = {"state": {"amount": 10000}, "questions": {"risk": {"type": "noul", "instructions": "Risk?"}}}
    call = {"name": tool.name, "type": "tool_call", "id": "legacy_call", "args": payload}
    config = {"tags": ["caller-tag"], "callbacks": [callback]}
    if asynchronous:

        async def run():
            async with classifier:
                return await tool.ainvoke(call, config)

        result = asyncio.run(run())
    else:
        with classifier:
            result = tool.invoke(call, config)
    assert result.tool_call_id == "legacy_call"
    assert json.loads(result.content)["request_id"] == "req_legacy"
    assert sent[0]["data"] == {"amount": 10000} and "state" not in sent[0]
    assert sent[0]["model"] == ("grayson-test" if conversion in {"bound", "combined"} else "grayson-1")
    assert "caller-tag" in callback.tags
    if conversion in {"configured", "combined"}:
        assert "bound-tag" in callback.tags
    assert payload["state"] == {"amount": 10000} and "data" not in payload
    assert "state" not in tool.args_schema.model_json_schema()["properties"]


def test_bound_tool_serialization_preserves_conversion():
    with FinicClassifier(api_key="do-not-store") as classifier:
        bound = classifier.bind(model="grayson-test").with_config(tags=["review"])
        payload = dumps(bound)
        assert "do-not-store" not in payload
        options = (
            {"allowed_objects": [type(bound), FinicClassifier]}
            if "allowed_objects" in inspect.signature(loads).parameters
            else {}
        )
        restored = loads(
            payload,
            valid_namespaces=["langchain_finic"],
            secrets_map={"FINIC_API_KEY": "replacement"},
            **options,
        )
        try:
            restored.bound.transport = httpx.MockTransport(response)
            result = restored.as_tool().invoke(
                {"state": "evidence", "questions": {"risk": {"type": "noul", "instructions": "Risk?"}}}
            )
            assert result["model"] == "grayson-test"
            assert restored.config["tags"] == ["review"]
        finally:
            restored.bound.close()
