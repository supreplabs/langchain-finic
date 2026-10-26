"""Real create_agent loops with mocked providers; no credentials or network access."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, ValidationError

from langchain_finic import DecisionRequest, FinicClassifier, FinicError
from langchain_finic.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware


class Chat(BaseChatModel):
    label: str
    calls: list = Field(default_factory=list)
    call_tool: bool = False

    @property
    def _llm_type(self):
        return "middleware-test-chat"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        message = AIMessage(content=self.label)
        if self.call_tool and isinstance(messages[-1], HumanMessage):
            message = AIMessage(
                content="", tool_calls=[{"name": "protected", "args": {"value": "demo"}, "id": "call_demo"}]
            )
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


def wire_response(request, *, choice="fast", probability=0.95, risk=0.1):
    answers = {}
    for name, q in request.questions.items():
        if q.type == "choice":
            k = len(q.criteria)
            answers[name] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    key: probability if key == choice else (1 - probability) / (k - 1) for key in q.criteria
                },
                "confidence": (k * probability - 1) / (k - 1),
            }
        else:
            answers[name] = {"type": "noul", "noul": risk}
    return {
        "id": "decision",
        "model": request.model,
        "request_id": "request",
        "evaluation_time_ms": 1,
        "usage": {"input_tokens": 20, "output_tokens": 0},
        "answers": answers,
    }


def choices(fast=None, powerful=None):
    return {
        "fast": ModelChoice(model=fast or Chat(label="fast"), criteria="Simple request"),
        "powerful": ModelChoice(model=powerful or Chat(label="powerful"), criteria="Complex request"),
    }


def request_for_tool(name="protected"):
    return ToolCallRequest(
        tool_call={"name": name, "args": {"value": "demo"}, "id": "call_demo", "type": "tool_call"},
        tool=None,
        state={"messages": [HumanMessage(content="Please do it")]},
        runtime=None,
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_router_one_decision_per_run_and_model_sticky_through_tools(async_mode):
    calls = []

    def handle(req):
        request = DecisionRequest.model_validate_json(req.content)
        calls.append(request)
        return httpx.Response(200, json=wire_response(request, choice="powerful"))

    fast, powerful = Chat(label="fast"), Chat(label="powerful", call_tool=True)

    @tool
    def protected(value: str) -> str:
        """A harmless test tool."""
        return value

    async def arun():
        async with FinicClassifier(api_key="fake", async_transport=httpx.MockTransport(handle)) as classifier:
            router = ModelRouterMiddleware(choices=choices(fast, powerful), classifier=classifier)
            return await create_agent(fast, tools=[protected], middleware=[router]).ainvoke(
                {"messages": [HumanMessage(content="Complex task")]}
            )

    if async_mode:
        result = asyncio.run(arun())
    else:
        with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handle)) as classifier:
            router = ModelRouterMiddleware(choices=choices(fast, powerful), classifier=classifier)
            result = create_agent(fast, tools=[protected], middleware=[router]).invoke(
                {"messages": [HumanMessage(content="Complex task")]}
            )
    assert len(calls) == 1 and not fast.calls and len(powerful.calls) == 2
    assert result["messages"][-1].content == "powerful"
    assert result["finic_model_route"]["selected"] == "powerful"


def test_router_checkpoint_reclassifies_latest_human_each_invocation():
    seen = []

    def handler(req):
        request = DecisionRequest.model_validate_json(req.content)
        seen.append(request.state["message"]["content"])
        choice = "fast" if seen[-1] == "short" else "powerful"
        return httpx.Response(200, json=wire_response(request, choice=choice))

    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as classifier:
        models = choices()
        router = ModelRouterMiddleware(choices=models, classifier=classifier)
        agent = create_agent(models["fast"].model, middleware=[router], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "same-thread"}}
        assert (
            agent.invoke({"messages": [HumanMessage(content="short")]}, config)["messages"][-1].content
            == "fast"
        )
        assert (
            agent.invoke({"messages": [HumanMessage(content="complex")]}, config)["messages"][-1].content
            == "powerful"
        )
    assert seen == ["short", "complex"]


def test_router_concurrent_async_runs_do_not_share_route():
    async def handler(req):
        request = DecisionRequest.model_validate_json(req.content)
        choice = request.state["message"]["content"]
        await asyncio.sleep(0.01 if choice == "fast" else 0)
        return httpx.Response(200, json=wire_response(request, choice=choice))

    async def run():
        async with FinicClassifier(
            api_key="fake", async_transport=httpx.MockTransport(handler)
        ) as classifier:
            models = choices()
            router = ModelRouterMiddleware(choices=models, classifier=classifier)
            agent = create_agent(models["fast"].model, middleware=[router])
            results = await asyncio.gather(
                *(
                    agent.ainvoke({"messages": [HumanMessage(content=name)]})
                    for name in ["fast", "powerful"] * 3
                )
            )
            assert [r["messages"][-1].content for r in results] == ["fast", "powerful"] * 3

    asyncio.run(run())


@pytest.mark.parametrize("failure", [False, True])
def test_router_low_probability_and_error_fallback(failure):
    def handler(req):
        request = DecisionRequest.model_validate_json(req.content)
        return (
            httpx.Response(503)
            if failure
            else httpx.Response(200, json=wire_response(request, probability=0.55))
        )

    with FinicClassifier(
        api_key="fake", transport=httpx.MockTransport(handler), max_attempts=1
    ) as classifier:
        router = ModelRouterMiddleware(
            choices=choices(), classifier=classifier, minimum_probability=0.8, fallback="powerful"
        )
        result = router.before_agent({"messages": [HumanMessage(content="task")]}, None)
        assert result["finic_model_route"]["selected"] == "powerful"
        assert result["finic_model_route"]["fallback_used"]
        assert result["finic_model_route"]["assessment_failed"] == failure
        strict = ModelRouterMiddleware(choices=choices(), classifier=classifier, minimum_probability=0.8)
        with pytest.raises(FinicError):
            strict.before_agent({"messages": [HumanMessage(content="task")]}, None)


@pytest.mark.parametrize("risk,allowed", [(0.49, True), (0.5, False), (0.99, False)])
@pytest.mark.parametrize("async_mode", [False, True])
def test_tool_guard_actual_agent_loop(risk, allowed, async_mode):
    executed = []
    assessed = []

    @tool
    def protected(value: str) -> str:
        """A harmless tool recording whether it was executed."""
        executed.append(value)
        return value

    def handler(req):
        request = DecisionRequest.model_validate_json(req.content)
        assessed.append(request)
        return httpx.Response(200, json=wire_response(request, risk=risk))

    async def arun():
        async with FinicClassifier(
            api_key="fake", async_transport=httpx.MockTransport(handler)
        ) as classifier:
            guard = AutoModeMiddleware(tools=[protected], classifier=classifier)
            return await create_agent(
                Chat(label="done", call_tool=True), tools=[protected], middleware=[guard]
            ).ainvoke({"messages": [HumanMessage(content="Do demo")]})

    if async_mode:
        result = asyncio.run(arun())
    else:
        with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as classifier:
            guard = AutoModeMiddleware(tools=[protected], classifier=classifier)
            result = create_agent(
                Chat(label="done", call_tool=True), tools=[protected], middleware=[guard]
            ).invoke({"messages": [HumanMessage(content="Do demo")]})
    assert bool(executed) == allowed and len(assessed) == 1
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.tool_call_id == "call_demo"
    assert (tool_message.status == "error") != allowed
    assert assessed[0].state["tool_call"] == {"name": "protected", "args": {"value": "demo"}}


@pytest.mark.parametrize("malformed", [False, True])
def test_guard_fail_closed_and_redacts_provider_errors(malformed):
    def handler(req):
        return (
            httpx.Response(200, json={"private": "secret"})
            if malformed
            else httpx.Response(503, text="secret")
        )

    with FinicClassifier(
        api_key="fake", transport=httpx.MockTransport(handler), max_attempts=1
    ) as classifier:
        guard = AutoModeMiddleware(tools=["protected"], classifier=classifier)
        message = guard.wrap_tool_call(request_for_tool(), lambda _: pytest.fail("Must not execute"))
        assert message.status == "error" and "secret" not in str(message)
        assert message.artifact["assessment_failed"]


def test_unlisted_tool_skips_classifier_and_handler_errors_propagate():
    def handler(req):
        request = DecisionRequest.model_validate_json(req.content)
        return httpx.Response(200, json=wire_response(request, risk=0.1))

    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as classifier:
        guard = AutoModeMiddleware(tools=["protected"], classifier=classifier)
        assert guard.wrap_tool_call(request_for_tool("other"), lambda _: "executed") == "executed"

        def broken(_):
            raise RuntimeError("tool failed")

        with pytest.raises(RuntimeError, match="tool failed"):
            guard.wrap_tool_call(request_for_tool(), broken)


def test_custom_criteria_are_sent_and_borrowed_classifier_not_closed():
    seen = []

    def handler(req):
        request = DecisionRequest.model_validate_json(req.content)
        seen.append(request)
        return httpx.Response(200, json=wire_response(request))

    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as classifier:
        with AutoModeMiddleware(
            tools=["protected"],
            classifier=classifier,
            instructions="Custom policy",
            criteria={"true": "Unauthorized", "false": "Authorized"},
        ) as guard:
            assert guard.wrap_tool_call(request_for_tool(), lambda _: "ok") == "ok"
        assert not classifier._closed
    assert seen[0].questions["tool_risk"].instructions == "Custom policy"
    assert seen[0].questions["tool_risk"].criteria == {"true": "Unauthorized", "false": "Authorized"}


def test_async_cancellation_propagates_without_executing_tool():
    async def handler(req):
        raise asyncio.CancelledError()

    async def run():
        async with FinicClassifier(
            api_key="fake", async_transport=httpx.MockTransport(handler)
        ) as classifier:
            guard = AutoModeMiddleware(tools=["protected"], classifier=classifier)
            with pytest.raises(asyncio.CancelledError):
                await guard.awrap_tool_call(request_for_tool(), lambda _: pytest.fail("Do not execute"))

    asyncio.run(run())


def test_validation_and_model_request_not_mutated():
    with pytest.raises(ValueError):
        AutoModeMiddleware(tools=[])
    with pytest.raises(ValidationError):
        AutoModeMiddleware(tools=["x"], threshold=float("nan"))
    with pytest.raises(ValueError):
        ModelRouterMiddleware(choices=choices(), fallback="missing")
    with FinicClassifier(api_key="fake") as classifier:
        router = ModelRouterMiddleware(choices=choices(), classifier=classifier)
        request = ModelRequest(
            system_prompt=None,
            tool_choice=None,
            tools=[],
            response_format=None,
            model=Chat(label="original"),
            messages=[],
            state={"messages": [], "finic_model_route": {"selected": "powerful"}},
            runtime=SimpleNamespace(),
        )
        actual = router.wrap_model_call(request, lambda value: value)
        assert actual.model.label == "powerful" and request.model.label == "original"


def test_router_and_guard_compose_and_propagate_tracing():
    from langchain_core.callbacks import BaseCallbackHandler

    events = []

    class Recorder(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            if kwargs.get("metadata", {}).get("ls_provider") == "finic":
                events.append(kwargs)

    def handler(req):
        payload = DecisionRequest.model_validate_json(req.content)
        return httpx.Response(200, json=wire_response(payload, choice="powerful", risk=0.8))

    @tool
    def protected(value: str) -> str:
        """A tool that must be blocked in this test."""
        pytest.fail("Guard must prevent execution")

    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as classifier:
        models = choices(powerful=Chat(label="powerful", call_tool=True))
        router = ModelRouterMiddleware(choices=models, classifier=classifier)
        guard = AutoModeMiddleware(tools=[protected], classifier=classifier)
        agent = create_agent(models["fast"].model, tools=[protected], middleware=[router, guard])
        result = agent.invoke(
            {"messages": [HumanMessage(content="Perform task")]},
            {"tags": ["composition-test"], "callbacks": [Recorder()]},
        )
        assert result["messages"][-1].content == "powerful"
        assert result["messages"][-2].status == "error"
    assert len(events) == 2
    assert all("composition-test" in event["tags"] for event in events)


def test_async_guard_failure_and_unlisted_passthrough():
    async def handler(req):
        return httpx.Response(200, json={"malformed": True})

    async def run():
        async with FinicClassifier(
            api_key="fake", async_transport=httpx.MockTransport(handler)
        ) as classifier:
            guard = AutoModeMiddleware(tools=["protected"], classifier=classifier)

            async def execute(request):
                assert request.tool_call["name"] == "other"
                return "ok"

            blocked = await guard.awrap_tool_call(request_for_tool(), execute)
            assert blocked.status == "error" and blocked.artifact["assessment_failed"]
            assert await guard.awrap_tool_call(request_for_tool("other"), execute) == "ok"

    asyncio.run(run())


def test_owned_classifier_is_closed(monkeypatch):
    monkeypatch.setenv("FINIC_API_KEY", "fake")
    with AutoModeMiddleware(tools=["protected"]) as guard:
        assert not guard.classifier._closed
    assert guard.classifier._closed

    async def run():
        async with ModelRouterMiddleware(choices=choices()) as router:
            assert not router.classifier._closed
        assert router.classifier._closed

    asyncio.run(run())
