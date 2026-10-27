import asyncio
import json
import threading
import time

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import ConfigurableField
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_finic import FinicScorer, ScoringInput


def response(request):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "model_id": body["model_id"],
            "score_id": body["score_id"],
            "score": 200,
            "request_id": "req_test",
        },
    )


def value(i=0):
    return {"score_id": str(i), "data": {"amount": i}}


@pytest.mark.parametrize("binding", ["direct", "config", "bind", "both", "configurable"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_bound_tool_json_and_capabilities(binding, asynchronous):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return response(request)

    model = FinicScorer(
        model_id="scm_test",
        api_key="test",
        transport=httpx.MockTransport(handle),
        async_transport=httpx.MockTransport(handle),
    )
    wrapped = {
        "direct": model,
        "config": model.with_config(tags=["review"]),
        "bind": model.bind(),
        "both": model.bind().with_config(tags=["review"]).bind(),
        "configurable": model.configurable_fields(model_id=ConfigurableField(id="model")).with_config(
            configurable={"model": "scm_other"}
        ),
    }[binding]
    tool = wrapped.as_tool()
    assert tool.name == "finic_score"
    assert not {"update_schema", "skip_schema_validation"} & tool.args.keys()
    graph = (
        StateGraph(MessagesState)
        .add_node("tools", ToolNode([tool]))
        .add_edge(START, "tools")
        .add_edge("tools", END)
        .compile()
    )
    call = {"name": tool.name, "args": value(), "id": "call_test", "type": "tool_call"}

    async def run():
        async with model:
            return await graph.ainvoke({"messages": [AIMessage("", tool_calls=[call])]})

    if asynchronous:
        result = asyncio.run(run())
    else:
        with model:
            result = graph.invoke({"messages": [AIMessage("", tool_calls=[call])]})
    message = result["messages"][-1]
    assert message.status == "success"
    assert json.loads(message.content)["score"] == 200
    assert calls[0]["model_id"] == ("scm_other" if binding == "configurable" else "scm_test")


@pytest.mark.parametrize(
    "flag,option",
    [("update_schema", "allow_schema_update"), ("skip_schema_validation", "allow_schema_bypass")],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_scoring_tool_requires_runtime_opt_in(flag, option, asynchronous):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return response(request)

    model = FinicScorer(
        model_id="scm_test",
        api_key="test",
        transport=httpx.MockTransport(handle),
        async_transport=httpx.MockTransport(handle),
    )
    blocked = model.with_config(tags=["review"]).as_tool(args_schema=ScoringInput)
    allowed = model.bind().as_tool(**{option: True})
    assert flag in allowed.args and "description" in allowed.args[flag]
    payload = {**value(), flag: True}

    async def run():
        async with model:
            with pytest.raises(ValueError, match="opt-in"):
                await blocked.ainvoke(payload)
            assert not calls
            await allowed.ainvoke(payload)

    if asynchronous:
        asyncio.run(run())
    else:
        with model:
            with pytest.raises(ValueError, match="opt-in"):
                blocked.invoke(payload)
            assert not calls
            allowed.invoke(payload)
    assert len(calls) == 1 and calls[0][flag] is True


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("limit", [None, 2])
def test_completion_order_limits_and_failures(asynchronous, limit):
    active = peak = 0
    lock = threading.Lock()

    def enter():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)

    def leave(request):
        nonlocal active
        with lock:
            active -= 1
        if json.loads(request.content)["score_id"] == "0":
            return httpx.Response(409)
        return response(request)

    def sync(request):
        enter()
        time.sleep(0.02)
        return leave(request)

    async def async_handle(request):
        enter()
        await asyncio.sleep(0.02)
        return leave(request)

    model = FinicScorer(
        model_id="scm_test",
        api_key="test",
        transport=httpx.MockTransport(sync),
        async_transport=httpx.MockTransport(async_handle),
    )
    config = None if limit is None else [{"max_concurrency": limit} for _ in range(12)]

    async def run():
        async with model:
            assert [r async for r in model.abatch_as_completed([])] == []
            return [
                r
                async for r in model.abatch_as_completed(
                    [value(i) for i in range(12)], config, return_exceptions=True
                )
            ]

    if asynchronous:
        results = asyncio.run(run())
    else:
        with model:
            assert list(model.batch_as_completed([])) == []
            results = list(
                model.batch_as_completed([value(i) for i in range(12)], config, return_exceptions=True)
            )
    assert peak == (limit or 5)
    assert {i for i, _ in results} == set(range(12))
    assert isinstance(dict(results)[0], Exception)
    assert active == 0
