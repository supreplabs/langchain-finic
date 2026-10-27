import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from test_middleware import choices, request_for_tool, wire_response

from langchain_finic import DecisionRequest, FinicClassifier
from langchain_finic.middleware import AutoModeMiddleware, ModelRouterMiddleware


def test_tool_semantics_are_registered_and_names_only_remain_supported():
    def operation(value: str) -> str:
        return value

    read = StructuredTool.from_function(operation, name="protected", description="Read a record only.")
    write = StructuredTool.from_function(
        operation, name="protected", description="Permanently delete records."
    )
    with FinicClassifier(api_key="test") as classifier:
        one = AutoModeMiddleware(tools=[read], classifier=classifier)._request(request_for_tool())
        two = AutoModeMiddleware(tools=[write], classifier=classifier)._request(request_for_tool())
        assert one.data != two.data
        assert one.data["tool_definition"]["parameters"]["properties"]["value"]["type"] == "string"
        assert one.data["tool_call"] == two.data["tool_call"]
        assert (
            "tool_definition"
            not in AutoModeMiddleware(tools=["protected"], classifier=classifier)
            ._request(request_for_tool())
            .data
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_router_context_distinguishes_followups_and_receives_runtime(asynchronous):
    calls = []

    def handle(req):
        payload = DecisionRequest.model_validate_json(req.content)
        calls.append(payload.data)
        selected = "powerful" if "complex" in json.dumps(payload.data) else "fast"
        return httpx.Response(200, json=wire_response(payload, choice=selected))

    model = FinicClassifier(
        api_key="test", transport=httpx.MockTransport(handle), async_transport=httpx.MockTransport(handle)
    )

    def build(state, context):
        return {"history": state["messages"][-2:], "policy": context["policy"]}

    router = ModelRouterMiddleware(choices=choices(), classifier=model, context_builder=build)
    states = [
        {"messages": [HumanMessage(content=task), HumanMessage(content="Continue")]}
        for task in ["complex investigation", "simple arithmetic"]
    ]
    runtime = SimpleNamespace(context={"policy": "least costly capable model"})

    async def run():
        async with model:
            return await asyncio.gather(*(router.abefore_agent(state, runtime) for state in states))

    if asynchronous:
        results = asyncio.run(run())
    else:
        with model:
            results = [router.before_agent(state, runtime) for state in states]
    assert [r["finic_model_route"]["selected"] for r in results] == ["powerful", "fast"]
    assert all(c["policy"] == runtime.context["policy"] for c in calls)
    assert all(isinstance(s["messages"][0], HumanMessage) for s in states)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("invalid", ["oversized", "not_object", "nonfinite"])
def test_invalid_context_blocks_or_falls_back_without_http(asynchronous, invalid):
    def forbidden(_):
        pytest.fail("No request expected")

    builder = {
        "oversized": lambda s, c: {"history": "x" * 2000},
        "not_object": lambda s, c: [],
        "nonfinite": lambda s, c: {"value": float("inf")},
    }[invalid]
    classifier = FinicClassifier(
        api_key="test",
        transport=httpx.MockTransport(forbidden),
        async_transport=httpx.MockTransport(forbidden),
    )
    guard = AutoModeMiddleware(
        tools=["protected"], classifier=classifier, context_builder=builder, max_context_bytes=1000
    )
    router = ModelRouterMiddleware(
        choices=choices(),
        classifier=classifier,
        context_builder=builder,
        max_context_bytes=1000,
        fallback="powerful",
    )
    state = {"messages": [HumanMessage(content="Continue")]}

    async def run():
        async with classifier:
            blocked = await guard.awrap_tool_call(
                request_for_tool(), lambda r: pytest.fail("Must not execute")
            )
            routed = await router.abefore_agent(state, None)
            return blocked, routed

    if asynchronous:
        blocked, routed = asyncio.run(run())
    else:
        with classifier:
            blocked = guard.wrap_tool_call(request_for_tool(), lambda r: pytest.fail("Must not execute"))
            routed = router.before_agent(state, None)
    assert blocked.artifact["assessment_failed"] is True
    assert routed["finic_model_route"]["assessment_failed"] is True
    assert routed["finic_model_route"]["selected"] == "powerful"


def test_guard_selected_window_does_not_replace_tool_call():
    with FinicClassifier(api_key="test") as classifier:
        guard = AutoModeMiddleware(
            tools=["protected"],
            classifier=classifier,
            context_builder=lambda state, ctx: {
                "messages": state["messages"][-1:],
                "tool_call": {"name": "fake"},
            },
        )
        request = request_for_tool()
        request.state["messages"].insert(0, HumanMessage(content="x" * 100000))
        payload = guard._request(request)
        assert payload.data["tool_call"]["name"] == "protected"
        assert len(payload.data["application_context"]["messages"]) == 1
        assert len(payload.model_dump_json()) < 2000
        default = AutoModeMiddleware(tools=["protected"], classifier=classifier)
        with pytest.raises(ValueError, match="max_context_bytes"):
            default._request(request)


@pytest.mark.parametrize("limit", [0, -1, True, 2_000_001])
def test_context_limit_validation(limit):
    with pytest.raises(ValueError):
        AutoModeMiddleware(tools=["protected"], max_context_bytes=limit)


def test_v1_and_dictionary_tool_schemas():
    from pydantic.v1 import BaseModel

    class LegacyArguments(BaseModel):
        value: str

    def operation(value: str) -> str:
        return value

    schemas = [
        LegacyArguments,
        {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    ]
    with FinicClassifier(api_key="test") as classifier:
        for schema in schemas:
            tool = StructuredTool.from_function(
                operation, name="protected", description="Read only", args_schema=schema
            )
            guard = AutoModeMiddleware(tools=[tool], classifier=classifier)
            assert (
                guard._request(request_for_tool()).data["tool_definition"]["parameters"]["properties"][
                    "value"
                ]["type"]
                == "string"
            )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_router_missing_human_still_raises_with_fallback(asynchronous):
    with FinicClassifier(api_key="test") as classifier:
        router = ModelRouterMiddleware(choices=choices(), classifier=classifier, fallback="powerful")
        with pytest.raises(ValueError, match="human message"):
            if asynchronous:
                asyncio.run(router.abefore_agent({"messages": []}, None))
            else:
                router.before_agent({"messages": []}, None)
