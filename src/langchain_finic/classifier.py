"""Native LangChain decision Runnable for the Grayson probability API."""

import logging
from typing import Any, Self, cast

import httpx
from langchain_core.load.serializable import to_json_not_implemented
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, StructuredTool
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr, model_validator
from pydantic.json_schema import SkipJsonSchema

from langchain_finic._config import env_option
from langchain_finic.client import FinicClient
from langchain_finic.models import DecisionRequest, DecisionResponse, ModelName

logger = logging.getLogger(__name__)


class _DecisionTool(StructuredTool):
    """Normalize the legacy envelope before older LangChain filters parsed fields."""

    def _parse_input(self, tool_input: str | dict[str, Any], tool_call_id: str | None):
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

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "classifier"]


class FinicClassifier(_DecisionToolConversion, RunnableSerializable[DecisionRequest, DecisionResponse]):
    """Typed decision model with LCEL, tools, batching and native asynchronous I/O.

    Credentials/options resolve explicit values, then FINIC_* environment defaults.
    Use a context manager to reuse owned connection pools. Outside a context,
    owned pools live for one invocation (including configurable-field clones).
    An injected FinicClient remains caller-owned. Streaming yields one complete
    decision, not token deltas.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)
    model: ModelName = Field(default_factory=lambda: env_option("MODEL", "grayson-1"), validate_default=True)
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    timeout: float | None = Field(default=None, gt=30, le=180)
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    client: SkipJsonSchema[FinicClient | None] = Field(default=None, exclude=True, repr=False)
    http_client: SkipJsonSchema[httpx.Client | None] = Field(default=None, exclude=True, repr=False)
    http_async_client: SkipJsonSchema[httpx.AsyncClient | None] = Field(
        default=None, exclude=True, repr=False
    )
    transport: SkipJsonSchema[httpx.BaseTransport | None] = Field(default=None, exclude=True, repr=False)
    async_transport: SkipJsonSchema[httpx.AsyncBaseTransport | None] = Field(
        default=None, exclude=True, repr=False
    )
    _runtime_client: FinicClient = PrivateAttr()
    _closed: bool = PrivateAttr(default=False)
    _pool_connections: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def configure_client(self) -> Self:
        if self.client is not None:
            if any(
                value is not None
                for value in (
                    self.api_key,
                    self.timeout,
                    self.max_attempts,
                    self.http_client,
                    self.http_async_client,
                    self.transport,
                    self.async_transport,
                )
            ):
                raise ValueError("Pass either client or client options")
            self._runtime_client = self.client
        else:
            self._runtime_client = self._new_client()
            self.api_key = self._runtime_client.config.api_key
            self.timeout = self._runtime_client.config.timeout
            self.max_attempts = self._runtime_client.config.max_attempts
        return self

    def _new_client(self) -> FinicClient:
        return FinicClient(
            api_key=self.api_key,
            timeout=self.timeout,
            max_attempts=self.max_attempts,
            http_client=self.http_client,
            http_async_client=self.http_async_client,
            transport=self.transport,
            async_transport=self.async_transport,
        )

    @property
    def _client(self) -> FinicClient:
        return self._runtime_client

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "classifier"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "FINIC_API_KEY"}

    @property
    def lc_attributes(self) -> dict[str, Any]:
        return {"model": self.model}

    def to_json(self):
        # Recreating borrowed clients/transports from JSON would silently change behavior.
        if any(
            value is not None
            for value in (
                self.client,
                self.http_client,
                self.http_async_client,
                self.transport,
                self.async_transport,
            )
        ):
            return to_json_not_implemented(self)
        return super().to_json()

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

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any):
        return _DecisionBinding(bound=self, config=cast(RunnableConfig, {**(config or {}), **kwargs}))

    def bind(self, **kwargs: Any):
        return _DecisionBinding(bound=self, kwargs=kwargs)

    def close(self) -> None:
        if self.client is None:
            self._client.close()
        self._closed = True

    async def aclose(self) -> None:
        if self.client is None:
            await self._client.aclose()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("FinicClassifier is closed")
        self._pool_connections = True
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        await self.aclose()
