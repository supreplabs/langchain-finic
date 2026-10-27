"""LangChain Runnable for a custom Finic scoring model."""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import ensure_config, get_config_list
from langchain_core.tools import BaseTool
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel

from langchain_finic._runnable import FinicRunnable
from langchain_finic.models import Text
from langchain_finic.scoring_models import ScoringInput, ScoringRequest, ScoringResponse

logger = logging.getLogger(__name__)


class FinicScorer(FinicRunnable[ScoringInput, ScoringResponse]):
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

    def as_tool(
        self,
        args_schema: type[BaseModel] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_types: dict[str, type] | None = None,
    ) -> BaseTool:
        chain = self | RunnableLambda(lambda result: result.model_dump(mode="json"))
        return chain.as_tool(
            args_schema=args_schema,
            arg_types=arg_types,
            name=name or "finic_score",
            description=description
            or (
                "Score one record with a custom Finic model. "
                "Supply a unique score_id; writes are not retried."
            ),
        )
