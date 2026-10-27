"""Native LangChain decision Runnable for the Grayson probability API."""

import logging
from typing import Any, Self, cast

from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableLambda,
)
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from langchain_finic._config import env_option
from langchain_finic._runnable import FinicRunnable
from langchain_finic.models import DecisionRequest, DecisionResponse, ModelName

logger = logging.getLogger(__name__)


class _DecisionTool(StructuredTool):
    """Normalize the legacy envelope before older LangChain filters parsed fields."""

    def _parse_input(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> str | dict[str, Any]:
        if isinstance(tool_input, dict) and "state" in tool_input:
            # Run the same conflict/null checks as direct invocation before renaming.
            DecisionRequest.model_validate(tool_input)
            tool_input = dict(tool_input)
            tool_input["data"] = tool_input.pop("state")
        return super()._parse_input(tool_input, tool_call_id)


def _decision_tool(
    runnable: Runnable[DecisionRequest, DecisionResponse],
    args_schema: type[BaseModel] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool:
    # Standard conversion preserves callbacks, async support, and schema overrides.
    # JSON output also produces valid ToolMessage content for chat-model agents.
    chain = runnable | RunnableLambda(lambda result: result.model_dump(mode="json", exclude_none=True))
    tool = chain.as_tool(
        args_schema=args_schema,
        name=name or "finic_decide",
        arg_types=arg_types,
        description=description or "Evaluate fraud and risk evidence; return typed probabilities.",
    )

    if (
        (args_schema is None or args_schema is DecisionRequest)
        and arg_types is None
        and isinstance(tool, StructuredTool)
    ):
        return _DecisionTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=tool.func,
            coroutine=tool.coroutine,
        )
    return tool


class _DecisionToolConversion:
    def as_tool(
        self,
        args_schema: type[BaseModel] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_types: dict[str, type] | None = None,
    ) -> BaseTool:
        return _decision_tool(
            cast(Runnable[DecisionRequest, DecisionResponse], self),
            args_schema,
            name=name,
            description=description,
            arg_types=arg_types,
        )


class _DecisionBinding(_DecisionToolConversion, RunnableBinding[DecisionRequest, DecisionResponse]):
    """Keep decision tool conversion when configuration/kwargs are bound."""

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        return cast(Self, super().with_config(config, **kwargs))

    def bind(self, **kwargs: Any) -> Self:
        return cast(Self, super().bind(**kwargs))

    def invoke(
        self, input: DecisionRequest | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> DecisionResponse:
        return super().invoke(cast(DecisionRequest, input), config, **kwargs)

    async def ainvoke(
        self, input: DecisionRequest | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> DecisionResponse:
        return await super().ainvoke(cast(DecisionRequest, input), config, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "classifier"]


class FinicClassifier(_DecisionToolConversion, FinicRunnable[DecisionRequest, DecisionResponse]):
    """Typed decision model with LCEL, tools, batching and native asynchronous I/O.

    Credentials/options resolve explicit values, then FINIC_* environment defaults.
    Use a context manager to reuse owned connection pools. Outside a context,
    owned pools live for one invocation (including configurable-field clones).
    An injected FinicClient remains caller-owned. Streaming yields one complete
    decision, not token deltas.
    """

    model: ModelName = Field(default_factory=lambda: env_option("MODEL", "grayson-1"), validate_default=True)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "classifier"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        return {"model": self.model}

    def _request(self, value: DecisionRequest | dict, model: str | None) -> DecisionRequest:
        if self._closed:
            raise RuntimeError("FinicClassifier is closed")
        data = value.model_dump() if isinstance(value, DecisionRequest) else dict(value)
        if isinstance(value, DecisionRequest) and "model" not in value.model_fields_set:
            data.pop("model", None)
        if model is not None:
            data["model"] = model
        else:
            data.setdefault("model", self.model)
        return DecisionRequest.model_validate(data)

    def _trace_config(self, config: RunnableConfig | None, model: str | None) -> RunnableConfig:
        config = ensure_config(config)
        config["metadata"] = {
            **config.get("metadata", {}),
            "ls_provider": "finic",
            "ls_model_name": model or self.model,
            "finic_api": "decide",
        }
        return config

    @staticmethod
    def _record_usage(result: DecisionResponse) -> None:
        try:
            run = get_current_run_tree()
            if run is not None:
                run.extra.setdefault("metadata", {}).update(
                    usage_metadata=result.usage_metadata,
                    finic_request_id=result.request_id,
                    finic_evaluation_time_ms=result.evaluation_time_ms,
                )
        except Exception:
            # Optional telemetry must never turn a successful decision into a failure.
            logger.debug("Unable to attach Finic usage metadata")

    def _invoke(self, value: DecisionRequest | dict, *, model: str | None = None) -> DecisionResponse:
        request = self._request(value, model)
        if self.client is not None or self._pool_connections:
            result = self._client.decide(request)
        else:
            # LangChain reconstructs configurable models without entering/exiting
            # their context. Never leave an unowned pool on those temporary models.
            with self._new_client() as client:
                result = client.decide(request)
        self._record_usage(result)
        return result

    async def _ainvoke(self, value: DecisionRequest | dict, *, model: str | None = None) -> DecisionResponse:
        request = self._request(value, model)
        if self.client is not None or self._pool_connections:
            result = await self._client.adecide(request)
        else:
            async with self._new_client() as client:
                result = await client.adecide(request)
        self._record_usage(result)
        return result

    def invoke(
        self,
        input: DecisionRequest | dict,
        config: RunnableConfig | None = None,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> DecisionResponse:
        if kwargs:
            raise TypeError(f"Unsupported Finic invocation options: {', '.join(sorted(kwargs))}")
        request = self._request(input, model)
        return self._call_with_config(
            self._invoke,
            request,
            self._trace_config(config, request.model),
            run_type="llm",
            model=model,
        )

    async def ainvoke(
        self,
        input: DecisionRequest | dict,
        config: RunnableConfig | None = None,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> DecisionResponse:
        if kwargs:
            raise TypeError(f"Unsupported Finic invocation options: {', '.join(sorted(kwargs))}")
        request = self._request(input, model)
        return await self._acall_with_config(
            self._ainvoke,
            request,
            self._trace_config(config, request.model),
            run_type="llm",
            model=model,
        )

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> _DecisionBinding:
        return _DecisionBinding(bound=self, config=cast(RunnableConfig, {**(config or {}), **kwargs}))

    def bind(self, **kwargs: Any) -> _DecisionBinding:
        return _DecisionBinding(bound=self, kwargs=kwargs)
