import asyncio
import json

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumps
from langchain_core.runnables import ConfigurableField
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


def decision_runnable(classifier, variant):
    if variant == "configured":
        return classifier.with_config(tags=["model-precedence"]), "constructor-model"
    if variant == "bound":
        return classifier.bind(model="bound-model"), "bound-model"
    if variant == "configurable":
        return classifier.configurable_fields(model=ConfigurableField(id="chosen_model")).with_config(
            configurable={"chosen_model": "configurable-model"}
        ), "configurable-model"
    if variant == "chained":
        return classifier.with_config(tags=["model-precedence"]).configurable_fields(
            model=ConfigurableField(id="chosen_model")
        ).with_config(configurable={"chosen_model": "configurable-model"}), "configurable-model"
    return classifier, "constructor-model"


@pytest.mark.parametrize("variant", ["direct", "configured", "bound", "configurable", "chained"])
@pytest.mark.parametrize("explicit_model", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_tool_preserves_model_precedence(monkeypatch, variant, explicit_model, asynchronous):
    monkeypatch.setenv("FINIC_MODEL", "env-model")
    payload = {
        "state": "fictional evidence",
        "questions": {"risk": {"type": "noul", "instructions": "Risk?"}},
    }
    if explicit_model:
        # Explicitly supplying the schema's default must remain an override.
        payload["model"] = "env-model"

    async def run_async():
        async with httpx.AsyncClient(transport=httpx.MockTransport(echo)) as http:
            async with FinicClassifier(
                api_key="fake", model="constructor-model", http_async_client=http
            ) as classifier:
                runnable, default = decision_runnable(classifier, variant)
                result = await runnable.as_tool(name="review").ainvoke(
                    {"name": "review", "args": payload, "id": "model-call", "type": "tool_call"}
                )
                return result, default

    if asynchronous:
        message, default = asyncio.run(run_async())
    else:
        with httpx.Client(transport=httpx.MockTransport(echo)) as http:
            with FinicClassifier(api_key="fake", model="constructor-model", http_client=http) as classifier:
                runnable, default = decision_runnable(classifier, variant)
                message = runnable.as_tool(name="review").invoke(
                    {"name": "review", "args": payload, "id": "model-call", "type": "tool_call"}
                )
    expected = "bound-model" if variant == "bound" else "env-model" if explicit_model else default
    assert message.tool_call_id == "model-call"
    assert json.loads(message.content)["model"] == expected
