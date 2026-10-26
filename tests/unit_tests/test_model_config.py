import asyncio
import json

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumps
from pydantic import ValidationError

from langchain_finic import DecisionRequest, DecisionResponse, FinicClassifier, Noul


def request(**kwargs):
    return DecisionRequest(state="example", questions={"risk": Noul(instructions="Risk?")}, **kwargs)


def echo(req):
    parsed = DecisionRequest.model_validate_json(req.content)
    return httpx.Response(
        200,
        json={
            "id": "test",
            "request_id": "req",
            "model": parsed.model,
            "answers": {"risk": {"type": "noul", "noul": 0.1}},
            "usage": {"input_tokens": 1, "output_tokens": 0},
            "evaluation_time_ms": 0,
        },
    )


def test_model_defaults_and_env(monkeypatch):
    monkeypatch.delenv("FINIC_MODEL", raising=False)
    monkeypatch.setenv("GRAYSON_MODEL", "ignored")
    assert request().model == "grayson-1"
    with FinicClassifier(api_key="fake") as classifier:
        assert classifier.model == "grayson-1"
    monkeypatch.setenv("FINIC_MODEL", "configured-model")
    assert request().model == "configured-model"
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(echo)) as classifier:
        assert classifier.invoke(request()).model == "configured-model"
        assert "configured-model" in dumps(FinicClassifier(api_key="fake"))


def test_constructor_request_and_bound_precedence(monkeypatch):
    monkeypatch.setenv("FINIC_MODEL", "env-model")
    with FinicClassifier(
        api_key="fake", model="constructor-model", transport=httpx.MockTransport(echo)
    ) as classifier:
        assert classifier.invoke(request()).model == "constructor-model"
        assert (
            classifier.invoke({"state": "text", "questions": {"risk": Noul(instructions="Risk?")}}).model
            == "constructor-model"
        )
        assert classifier.invoke(request(model="request-model")).model == "request-model"
        assert (
            classifier.bind(model="bound-model").invoke(request(model="request-model")).model == "bound-model"
        )
        assert classifier.invoke(request(model="request-model"), model="call-model").model == "call-model"


def test_async_model_and_trace_use_actual_request(monkeypatch):
    monkeypatch.setenv("FINIC_MODEL", "env-model")

    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.models = []

        def on_llm_start(self, serialized, prompts, **kwargs):
            self.models.append(kwargs["metadata"]["ls_model_name"])

    # The custom Runnable emits its configured LLM run through the generic start hook.
    class ChainRecorder(Recorder):
        def on_chain_start(self, serialized, inputs, **kwargs):
            self.models.append(kwargs["metadata"]["ls_model_name"])

    recorder = ChainRecorder()

    async def run():
        async with FinicClassifier(api_key="fake", async_transport=httpx.MockTransport(echo)) as classifier:
            result = await classifier.ainvoke(request(model="request-model"), {"callbacks": [recorder]})
            assert result.model == "request-model"

    asyncio.run(run())
    assert recorder.models and set(recorder.models) == {"request-model"}


@pytest.mark.parametrize("model", ["", " ", "invalid model"])
def test_invalid_model_rejected(monkeypatch, model):
    monkeypatch.setenv("FINIC_MODEL", model)
    with pytest.raises(ValidationError):
        request()
    with pytest.raises(ValidationError):
        FinicClassifier(api_key="fake")


def test_response_model_must_match_request():
    body = json.loads(
        echo(
            httpx.Request("POST", "https://example.test", content=request(model="one").model_dump_json())
        ).content
    )
    body["model"] = "two"
    with pytest.raises(ValueError, match="model"):
        DecisionResponse.model_validate(body).validate_for(request(model="one"))
