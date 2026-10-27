"""LangChain Runnable for a custom Finic scoring model."""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Literal, cast, overload

from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_core.runnables.config import ensure_config, get_config_list
from langchain_core.runnables.utils import AnyConfigurableField
from langchain_core.tools import BaseTool, StructuredTool
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field, StrictBool, TypeAdapter, create_model

from langchain_finic._runnable import FinicRunnable
from langchain_finic.models import Text
from langchain_finic.scoring_models import ScoringInput, ScoringRequest, ScoringResponse

logger = logging.getLogger(__name__)


class _ScoringTool(StructuredTool):
    """Enforce application-selected capabilities even for manually supplied arguments."""

    disabled_flags: frozenset[str] = Field(exclude=True)

    def _parse_input(self, tool_input: str | dict[str, Any], tool_call_id: str | None):
        def check(value, *, parsed=False):
            if isinstance(value, dict) and any(
                key in value and (not parsed or value[key] is not None) for key in self.disabled_flags
            ):
                raise ValueError("Scoring tool schema changes/bypass require explicit construction opt-in")

        check(tool_input)
        parsed = super()._parse_input(tool_input, tool_call_id)
        check(parsed, parsed=True)
        return parsed


class _ScorerToolConversion:
    def as_tool(
        self,
        args_schema: type[BaseModel] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_types: dict[str, type] | None = None,
        allow_schema_update: bool = False,
        allow_schema_bypass: bool = False,
    ) -> BaseTool:
        allow_schema_update = TypeAdapter(StrictBool).validate_python(allow_schema_update)
        allow_schema_bypass = TypeAdapter(StrictBool).validate_python(allow_schema_bypass)
        disabled = frozenset(
            key
            for key, enabled in (
                ("update_schema", allow_schema_update),
                ("skip_schema_validation", allow_schema_bypass),
            )
            if not enabled
        )
        if args_schema is None and arg_types is None:
            fields: dict[str, Any] = {
                key: (field.annotation or Any, field)
                for key, field in ScoringInput.model_fields.items()
                if key not in disabled
            }
            args_schema = create_model(
                "FinicScoringToolInput", __config__=ScoringInput.model_config, **fields
            )
        runnable = cast(Runnable[ScoringInput, ScoringResponse], self)
        chain = runnable | RunnableLambda(lambda result: result.model_dump(mode="json"))
        tool = chain.as_tool(
            args_schema=args_schema,
            arg_types=arg_types,
            name=name or "finic_score",
            description=description
            or (
                "Score one record with a custom Finic model. "
                "Supply a unique score_id; writes are not retried."
            ),
        )
        if not isinstance(tool, StructuredTool):
            raise TypeError("Scoring tools require an object argument schema")
        return _ScoringTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=tool.func,
            coroutine=tool.coroutine,
            disabled_flags=disabled,
        )


class _ScorerBinding(_ScorerToolConversion, RunnableBinding[ScoringInput, ScoringResponse]):
    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "scorer"]

    def configurable_fields(self, **kwargs: AnyConfigurableField) -> "_ScorerBinding":
        bound = cast(RunnableSerializable[ScoringInput, ScoringResponse], self.bound)
        return self.model_copy(update={"bound": bound.configurable_fields(**kwargs)})


class FinicScorer(_ScorerToolConversion, FinicRunnable[ScoringInput, ScoringResponse]):
    """Score records with a configured custom model using caller-provided score IDs.

    Calls make one HTTP attempt. Batches default to five concurrent calls; use
    max_concurrency to override. Concurrency is not an organization rate limiter.
    Outside a context, owned clients close after each call. Injected clients are
    caller-owned. Streaming emits one complete result, not score/token deltas.
    """

    model_id: Text

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "scorer"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        return {"model_id": self.model_id}

    def _input(self, value: ScoringInput | dict) -> ScoringInput:
        if self._closed:
            raise RuntimeError("FinicScorer is closed")
        # Revalidate even model instances; never let an input override the configured model.
        data = value.model_dump(exclude_none=True) if isinstance(value, ScoringInput) else value
        return ScoringInput.model_validate(data)

    def _trace_config(self, config: RunnableConfig | None) -> RunnableConfig:
        config = ensure_config(config)
        config["metadata"] = {
            **config.get("metadata", {}),
            "ls_provider": "finic",
            "finic_api": "score",
            "finic_model_id": self.model_id,
        }
        return config

    @staticmethod
    def _record_result(result: ScoringResponse) -> None:
        try:
            run = get_current_run_tree()
            if run is not None:
                run.extra.setdefault("metadata", {}).update(
                    finic_request_id=result.request_id,
                    finic_score_id=result.score_id,
                    finic_model_id=result.model_id,
                )
        except Exception:
            logger.debug("Unable to attach Finic scoring metadata")

    def _invoke(self, value: ScoringInput) -> ScoringResponse:
        request = ScoringRequest(model_id=self.model_id, **value.model_dump(exclude_none=True))
        if self.client is not None or self._pool_connections:
            result = self._client.score(request)
        else:
            with self._new_client() as client:
                result = client.score(request)
        self._record_result(result)
        return result

    async def _ainvoke(self, value: ScoringInput) -> ScoringResponse:
        request = ScoringRequest(model_id=self.model_id, **value.model_dump(exclude_none=True))
        if self.client is not None or self._pool_connections:
            result = await self._client.ascore(request)
        else:
            async with self._new_client() as client:
                result = await client.ascore(request)
        self._record_result(result)
        return result

    def invoke(
        self, input: ScoringInput | dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> ScoringResponse:
        if kwargs:
            raise TypeError(f"Unsupported scoring options: {', '.join(sorted(kwargs))}")
        return self._call_with_config(self._invoke, self._input(input), self._trace_config(config))

    async def ainvoke(
        self, input: ScoringInput | dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> ScoringResponse:
        if kwargs:
            raise TypeError(f"Unsupported scoring options: {', '.join(sorted(kwargs))}")
        return await self._acall_with_config(self._ainvoke, self._input(input), self._trace_config(config))

    @staticmethod
    def _batch_configs(config, count):
        configs = get_config_list(config, count)
        for item in configs:
            if item.get("max_concurrency") is None:
                item["max_concurrency"] = 5
        return configs

    def batch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        return super().batch(
            inputs, self._batch_configs(config, len(inputs)), return_exceptions=return_exceptions, **kwargs
        )

    async def abatch(self, inputs, config=None, *, return_exceptions=False, **kwargs):
        return await super().abatch(
            inputs, self._batch_configs(config, len(inputs)), return_exceptions=return_exceptions, **kwargs
        )

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[ScoringInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, ScoringResponse]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[ScoringInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, ScoringResponse | Exception]]: ...

    def batch_as_completed(
        self, inputs: Sequence[ScoringInput], config=None, *, return_exceptions=False, **kwargs: Any
    ) -> Iterator[tuple[int, ScoringResponse | Exception]]:
        yield from super().batch_as_completed(
            inputs, self._batch_configs(config, len(inputs)), return_exceptions=return_exceptions, **kwargs
        )

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[ScoringInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, ScoringResponse]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[ScoringInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, ScoringResponse | Exception]]: ...

    async def abatch_as_completed(
        self, inputs: Sequence[ScoringInput], config=None, *, return_exceptions=False, **kwargs: Any
    ) -> AsyncIterator[tuple[int, ScoringResponse | Exception]]:
        configs = self._batch_configs(config, len(inputs))
        if not inputs:
            return
        limit = configs[0]["max_concurrency"]
        if limit < 1:
            raise ValueError("max_concurrency must be positive")
        semaphore = asyncio.Semaphore(limit)

        async def run(index: int) -> tuple[int, ScoringResponse | Exception]:
            async with semaphore:
                try:
                    result = await self.ainvoke(inputs[index], configs[index], **kwargs)
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    return index, exc
                return index, result

        tasks = [asyncio.create_task(run(index)) for index in range(len(inputs))]
        try:
            for completed in asyncio.as_completed(tasks):
                yield await completed
        finally:
            # Do not leave queued writes alive after cancellation, closure or failure.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> _ScorerBinding:
        return _ScorerBinding(bound=self, config=cast(RunnableConfig, {**(config or {}), **kwargs}))

    def bind(self, **kwargs: Any) -> _ScorerBinding:
        return _ScorerBinding(bound=self, kwargs=kwargs)

    def configurable_fields(self, **kwargs: AnyConfigurableField) -> _ScorerBinding:
        return _ScorerBinding(bound=super().configurable_fields(**kwargs))
