"""Consumer type assertions. Checked by make type_check; never executed."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, assert_type

from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.runnables import ConfigurableField, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import Command

from langchain_finic import (
    DecisionRequest,
    DecisionResponse,
    FinicClassifier,
    FinicScorer,
    ScoringInput,
    ScoringResponse,
)
from langchain_finic.middleware import AutoModeMiddleware, ModelRouterMiddleware, ModelRouterState


def scoring(scorer: FinicScorer, record: ScoringInput, data: dict[str, Any], config: RunnableConfig) -> None:
    assert_type(scorer.invoke(data, config), ScoringResponse)
    assert_type(scorer.batch([record]), list[ScoringResponse])
    assert_type(scorer.batch([data], return_exceptions=True), list[ScoringResponse | Exception])
    assert_type(scorer.batch_as_completed([data]), Iterator[tuple[int, ScoringResponse]])
    assert_type(
        scorer.batch_as_completed([record], return_exceptions=True),
        Iterator[tuple[int, ScoringResponse | Exception]],
    )
    assert_type(scorer.abatch_as_completed([data]), AsyncIterator[tuple[int, ScoringResponse]])
    assert_type(scorer.with_config(config).invoke(data), ScoringResponse)
    assert_type(scorer.with_config(config).as_tool(), BaseTool)
    assert_type(scorer.bind().with_config(config).bind().as_tool(allow_schema_update=True), BaseTool)
    assert_type(
        scorer.configurable_fields(model_id=ConfigurableField(id="model"))
        .with_config(config)
        .as_tool(allow_schema_bypass=True),
        BaseTool,
    )
    with scorer as entered:
        assert_type(entered, FinicScorer)


async def async_scoring(scorer: FinicScorer, data: dict[str, Any]) -> None:
    assert_type(await scorer.ainvoke(data), ScoringResponse)
    assert_type(await scorer.abatch([data]), list[ScoringResponse])
    assert_type(await scorer.abatch([data], return_exceptions=True), list[ScoringResponse | Exception])
    async with scorer as entered:
        assert_type(entered, FinicScorer)


def decisions(classifier: FinicClassifier, request: DecisionRequest, data: dict[str, Any]) -> None:
    assert_type(classifier.invoke(data), DecisionResponse)
    assert_type(classifier.with_config(tags=["typing"]).invoke(request), DecisionResponse)
    assert_type(classifier.bind().as_tool(), BaseTool)
    assert_type(classifier.bind().with_config(tags=["typing"]).invoke(data), DecisionResponse)
    with classifier as entered:
        assert_type(entered, FinicClassifier)


def middleware(
    router: ModelRouterMiddleware,
    guard: AutoModeMiddleware,
    state: AgentState,
    runtime: Runtime[Any],
    model_request: ModelRequest,
    tool_request: ToolCallRequest,
) -> None:
    def model_handler(request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def tool_handler(request: ToolCallRequest) -> ToolMessage | Command[Any]:
        raise NotImplementedError

    assert_type(router.before_agent(state, runtime), dict[str, Any])
    assert_type(router.wrap_model_call(model_request, model_handler), ModelResponse)
    assert_type(guard.wrap_tool_call(tool_request, tool_handler), ToolMessage | Command[Any])


async def async_middleware(
    router: ModelRouterMiddleware,
    guard: AutoModeMiddleware,
    state: AgentState,
    runtime: Runtime[Any],
    model_request: ModelRequest,
    tool_request: ToolCallRequest,
) -> None:
    async def model_handler(request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    async def tool_handler(request: ToolCallRequest) -> ToolMessage | Command[Any]:
        raise NotImplementedError

    assert_type(await router.abefore_agent(state, runtime), dict[str, Any])
    assert_type(await router.awrap_model_call(model_request, model_handler), ModelResponse)
    assert_type(await guard.awrap_tool_call(tool_request, tool_handler), ToolMessage | Command[Any])


def dynamic_exception_flag(scorer: FinicScorer, data: dict[str, Any], flag: bool) -> None:
    assert_type(scorer.batch([data], return_exceptions=flag), list[ScoringResponse | Exception])
    assert_type(
        scorer.batch_as_completed([data], return_exceptions=flag),
        Iterator[tuple[int, ScoringResponse | Exception]],
    )


def route_state(state: ModelRouterState) -> None:
    if "finic_model_route" in state:
        assert_type(state["finic_model_route"]["selected"], str)
        assert_type(state["finic_model_route"]["fallback_used"], bool)
