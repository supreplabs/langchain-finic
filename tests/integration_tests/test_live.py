"""Explicit opt-in: eighteen paid calls, no retries or customer data."""

import asyncio
import json
import os

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as make_tool
from langchain_tests.integration_tests.tools import ToolsIntegrationTests
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_finic import (
    Choice,
    DecisionRequest,
    DecisionResponse,
    FinicClassifier,
    MultiChoice,
    Noul,
    Score,
)
from langchain_finic.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("FINIC_LIVE_TESTS") != "1" or not os.getenv("FINIC_API_KEY"),
        reason="Set FINIC_LIVE_TESTS=1 and export FINIC_API_KEY to spend API credits",
    ),
]


def request():
    return DecisionRequest(
        state={"events": ["Fictional customer uses their usual device for a routine payment"]},
        questions={
            "risk": Noul(instructions="Does the evidence suggest fraud?"),
            "action": Choice(
                instructions="Choose a review action.",
                criteria={"routine": "Routine review", "investigate": "Investigate further"},
            ),
            "signals": MultiChoice(
                instructions="Which indicators are supported by the evidence?",
                criteria={"usual_device": "Usual device", "new_payee": "New payee"},
            ),
            "severity": Score(instructions="Assess concern.", criteria=["Low", "Medium", "High"]),
        },
    )


def assert_decision(result, expected):
    result.validate_for(expected)
    assert result.request_id
    assert len(result.nouls) == len(result.choices) == len(result.multi_choices) == len(result.scores) == 1


class TestFinicToolLive(ToolsIntegrationTests):
    """Official tool contracts against Finic: four calls plus one offline check."""

    @property
    def tool_constructor(self) -> BaseTool:
        # Invocation-scoped clients close after each sync or async tool call.
        return FinicClassifier(max_attempts=1).as_tool()

    @property
    def tool_invoke_params_example(self) -> dict:
        return request().model_dump(mode="json")


def test_live_sync():
    payload = request()
    with FinicClassifier(max_attempts=1) as classifier:
        result = classifier.invoke(payload)
    assert_decision(result, payload)


def test_live_async():
    payload = request()

    async def run():
        async with FinicClassifier(max_attempts=1) as classifier:
            return await classifier.ainvoke(payload)

    result = asyncio.run(run())
    assert_decision(result, payload)


@pytest.mark.parametrize("async_mode", [False, True])
def test_live_batch(async_mode):
    inputs = [
        request().model_copy(
            update={"questions": {f"{key}_{i}": q for key, q in request().questions.items()}}
        )
        for i in range(2)
    ]

    async def run():
        async with FinicClassifier(max_attempts=1) as classifier:
            return await classifier.abatch(inputs, {"max_concurrency": 2})

    if async_mode:
        results = asyncio.run(run())
    else:
        with FinicClassifier(max_attempts=1) as classifier:
            results = classifier.batch(inputs, {"max_concurrency": 2})
    assert len(results) == len(inputs)
    for result, payload in zip(results, inputs, strict=True):
        assert_decision(result, payload)


@pytest.mark.parametrize("async_mode", [False, True])
def test_live_langgraph_tool(async_mode):
    payload = request()

    def graph_for(classifier):
        finic_tool = classifier.as_tool()
        graph = (
            StateGraph(MessagesState)
            .add_node("tools", ToolNode([finic_tool]))
            .add_edge(START, "tools")
            .add_edge("tools", END)
            .compile()
        )
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": finic_tool.name,
                            "args": payload.model_dump(mode="json"),
                            "id": "live_decision",
                        }
                    ],
                )
            ]
        }
        return graph, state

    async def run():
        async with FinicClassifier(max_attempts=1) as classifier:
            graph, state = graph_for(classifier)
            return await graph.ainvoke(state)

    if async_mode:
        result = asyncio.run(run())
    else:
        with FinicClassifier(max_attempts=1) as classifier:
            graph, state = graph_for(classifier)
            result = graph.invoke(state)
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage) and message.status == "success"
    assert message.tool_call_id == "live_decision"
    assert_decision(DecisionResponse.model_validate(json.loads(message.content)), payload)


class ScriptedChat(FakeMessagesListChatModel):
    """Only the host chat model is scripted; Finic decisions use the live API."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("async_mode", [False, True])
def test_live_router_and_guard_agent(async_mode):
    executed = []

    @make_tool
    def echo_demo(value: str) -> str:
        """Echo a synthetic value locally; no external side effects."""
        executed.append(value)
        return value

    def agent_for(classifier):
        def chat(label):
            return ScriptedChat(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "echo_demo",
                                "args": {"value": "synthetic-demo"},
                                "id": "live_echo",
                            }
                        ],
                    ),
                    AIMessage(content=label),
                ]
            )

        choices = {
            "fast": ModelChoice(model=chat("fast"), criteria="Simple echo requests"),
            "powerful": ModelChoice(model=chat("powerful"), criteria="Complex analytical requests"),
        }
        router = ModelRouterMiddleware(choices=choices, classifier=classifier)
        guard = AutoModeMiddleware(tools=[echo_demo], classifier=classifier)
        return create_agent(choices["fast"].model, tools=[echo_demo], middleware=[router, guard])

    state = {"messages": [HumanMessage(content="Please echo synthetic-demo using the local echo tool.")]}

    async def run():
        async with FinicClassifier(max_attempts=1) as classifier:
            return await agent_for(classifier).ainvoke(state, {"recursion_limit": 6})

    if async_mode:
        result = asyncio.run(run())
    else:
        with FinicClassifier(max_attempts=1) as classifier:
            result = agent_for(classifier).invoke(state, {"recursion_limit": 6})
    route = result["finic_model_route"]
    assert route["selected"] in {"fast", "powerful"}
    assert not route["assessment_failed"] and not route["fallback_used"]
    assert result["messages"][-1].content == route["selected"]
    tool_message = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert tool_message.tool_call_id == "live_echo"
    if tool_message.status == "error":
        assert not executed
        assert tool_message.artifact["assessment_failed"] is False
        assert 0.5 <= tool_message.artifact["risk_probability"] <= 1
    else:
        assert executed == ["synthetic-demo"]
        assert tool_message.content == "synthetic-demo"


@pytest.mark.parametrize(
    "evidence",
    [
        {"data": {"amount": 10}, "context": "Fictional routine payment"},
        {"context": "Fictional customer making a routine payment from their usual device"},
    ],
)
def test_live_documented_inputs(evidence):
    request = DecisionRequest(
        **evidence, questions={"risk": Noul(instructions="Does the evidence suggest fraud?")}
    )
    with FinicClassifier(max_attempts=1) as classifier:
        result = classifier.invoke(request)
    result.validate_for(request)
    assert result.request_id
