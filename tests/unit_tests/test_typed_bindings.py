"""Binding typing forwards to LangChain without changing its runtime semantics."""

import asyncio
import json

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import ConfigurableField

from langchain_finic import FinicClassifier, FinicError, FinicScorer


@pytest.mark.parametrize("kind", ["decision", "scoring"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_bound_batches_keep_dict_inputs_errors_and_config(kind, asynchronous):
    calls = []
    tags = []

    class Recorder(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            tags.append(kwargs.get("tags", []))

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if body["data"]["fail"]:
            return httpx.Response(400)
        if kind == "scoring":
            output = {
                "model_id": body["model_id"],
                "score_id": body["score_id"],
                "score": 200,
                "request_id": "req_test",
            }
        else:
            output = {
                "id": "dec_test",
                "model": body["model"],
                "answers": {"risk": {"type": "noul", "noul": 0.2}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "request_id": "req_test",
                "evaluation_time_ms": 1,
            }
        return httpx.Response(200, json=output)

    options = {
        "api_key": "test",
        "max_attempts": 1,
        "transport": httpx.MockTransport(respond),
        "async_transport": httpx.MockTransport(respond),
    }
    if kind == "scoring":
        model = FinicScorer(model_id="scm_original", **options)
        runnable = model.configurable_fields(model_id=ConfigurableField(id="model"))
        base = {"score_id": "record_1"}
        configuration = {"configurable": {"model": "scm_selected"}}
    else:
        model = FinicClassifier(**options)
        runnable = model
        base = {"questions": {"risk": {"type": "noul", "instructions": "Risk?"}}}
        configuration = {}
    bound = runnable.bind().with_config(**configuration, tags=["bound"], callbacks=[Recorder()]).bind()
    records = [{**base, "data": {"fail": False}}, {**base, "data": {"fail": True}}]

    async def run():
        async with model:
            normal = await bound.abatch(records, return_exceptions=True)
            completed = [
                result async for result in bound.abatch_as_completed(records, return_exceptions=True)
            ]
            with pytest.raises(FinicError):
                await bound.abatch(records[1:])
            return normal, completed

    if asynchronous:
        normal, completed = asyncio.run(run())
    else:
        with model:
            normal = bound.batch(records, return_exceptions=True)
            completed = list(bound.batch_as_completed(records, return_exceptions=True))
            with pytest.raises(FinicError):
                bound.batch(records[1:])
    assert normal[0].request_id == "req_test"
    assert isinstance(normal[1], FinicError)
    assert dict(completed)[0].request_id == "req_test"
    assert isinstance(dict(completed)[1], FinicError)
    assert len(calls) == 5
    assert all("bound" in values for values in tags)
    assert tags
    if kind == "scoring":
        assert all(body["model_id"] == "scm_selected" for body in calls)
