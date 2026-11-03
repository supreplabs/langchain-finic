"""Optional LangChain adapters for the Finic Kairos Python SDK.

Install ``langchain-finic[kairos]`` for the SDK. The adapter borrows a caller-owned
client; construct and shut down that client in the application lifecycle.
"""

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal, Protocol, Self, cast

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

KairosOutcome = Literal["approve", "step_up", "review", "block"]
KairosOperation = Literal["decide", "track", "track_and_decide", "run_workflow", "flush"]


class KairosClientProtocol(Protocol):
    """The synchronous portion of ``finic-kairos`` used by these adapters."""

    def decide(self, action: str, properties: Mapping[str, Any] | None = None, **kwargs: Any) -> Any: ...

    def track(self, event_type: str, properties: Mapping[str, Any] | None = None, **kwargs: Any) -> None: ...

    def track_and_decide(
        self, event_type: str, properties: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Any: ...

    def run_workflow(
        self, name: str, input: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]: ...

    def flush(self) -> None: ...


class KairosDecisionInput(BaseModel):
    """A Kairos action and the properties for the decision being requested."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    action: str = Field(min_length=1, description="The key of an action already defined in Kairos.")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Pending action properties; include an entity ID such as user_id."
    )
    session: str | None = None
    entities: dict[str, str] | None = None
    fallback: KairosOutcome | None = Field(
        default=None,
        description="Explicit outcome if Kairos cannot answer before timeout; omit to raise the SDK error.",
    )
    timeout: float | None = Field(default=None, gt=0)
    timestamp: str | datetime | None = Field(
        default=None, description="Optional ISO 8601 time or datetime for historical decisions."
    )
    flush: bool = True


class KairosEventInput(BaseModel):
    """An event to append to Kairos history."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    event_type: str = Field(min_length=1, description="The event type, such as login or payment.")
    properties: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | datetime | None = None
    event_id: str | None = None
    universal_id: str | None = None
    status: str | None = None
    session: str | None = None
    set: dict[str, Any] | None = None
    set_once: dict[str, Any] | None = None
    entities: dict[str, str] | None = None


class KairosTrackAndDecideInput(KairosEventInput):
    """Record the triggering event and request its action decision."""

    fallback: KairosOutcome | None = None
    timeout: float | None = Field(default=None, gt=0)
    flush: bool = True


class KairosWorkflowInput(BaseModel):
    """Start a named Kairos workflow with its manual-trigger input."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str = Field(min_length=1, description="The workflow name configured in Kairos.")
    input: dict[str, Any] = Field(default_factory=dict)
    wait: bool = False
    timeout_seconds: float | None = Field(default=None, ge=1, le=30)
    idempotency_key: str | None = None


class KairosDecisionOutput(BaseModel):
    """Validated, JSON-safe decision result returned by Kairos."""

    model_config = ConfigDict(extra="forbid")

    decision: KairosOutcome
    risk_score: int = Field(ge=0, le=100)
    probability: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    typology: str | None = None
    rules: list[str] = Field(default_factory=list)
    action: str | None = None
    decision_id: str | None = None
    event_id: str | None = None
    fallback: bool = False
    error: str | None = None


def _decision_output(decision: Any) -> KairosDecisionOutput:
    return KairosDecisionOutput(
        decision=decision.decision,
        risk_score=decision.risk_score,
        probability=decision.probability,
        reasons=decision.reasons,
        typology=decision.typology,
        rules=decision.rules,
        action=decision.action,
        decision_id=decision.decision_id,
        event_id=decision.event_id,
        fallback=decision.fallback,
        error=str(decision.error) if decision.error is not None else None,
    )


class KairosRunnable(RunnableSerializable[KairosDecisionInput, KairosDecisionOutput]):
    """Typed LangChain Runnable for asking Kairos to decide a pending action.

    The client is borrowed and is never serialized or shut down by this Runnable.
    The synchronous Kairos SDK call is run in a worker thread for ``ainvoke``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)

    client: Any = Field(exclude=True, repr=False)
    _kairos: KairosClientProtocol = PrivateAttr()

    @model_validator(mode="after")
    def set_client(self) -> Self:
        self._kairos = self.client
        return self

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_finic", "kairos"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def _input(self, value: KairosDecisionInput | dict[str, Any]) -> KairosDecisionInput:
        return KairosDecisionInput.model_validate(value)

    def _invoke(self, request: KairosDecisionInput) -> KairosDecisionOutput:
        arguments = request.model_dump(exclude={"action", "properties"}, exclude_none=True)
        result = self._kairos.decide(request.action, request.properties, **arguments)
        return _decision_output(result)

    async def _ainvoke(self, request: KairosDecisionInput) -> KairosDecisionOutput:
        return await asyncio.to_thread(self._invoke, request)

    @staticmethod
    def _trace_config(config: RunnableConfig | None, action: str) -> RunnableConfig:
        configured = ensure_config(config)
        configured["metadata"] = {
            **configured.get("metadata", {}),
            "ls_provider": "finic-kairos",
            "kairos_api": "decide",
            "kairos_action": action,
        }
        return configured

    def invoke(
        self, input: KairosDecisionInput | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> KairosDecisionOutput:
        if kwargs:
            raise TypeError(f"Unsupported Kairos invocation options: {', '.join(sorted(kwargs))}")
        request = self._input(input)
        return self._call_with_config(
            self._invoke, request, self._trace_config(config, request.action), run_type="chain"
        )

    async def ainvoke(
        self, input: KairosDecisionInput | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> KairosDecisionOutput:
        if kwargs:
            raise TypeError(f"Unsupported Kairos invocation options: {', '.join(sorted(kwargs))}")
        request = self._input(input)
        return await self._acall_with_config(
            self._ainvoke, request, self._trace_config(config, request.action), run_type="chain"
        )

    def as_tool(
        self,
        args_schema: type[BaseModel] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_types: dict[str, type] | None = None,
    ) -> BaseTool:
        """Convert the decision into a tool with JSON-safe structured output."""
        runnable = cast(Runnable[KairosDecisionInput, KairosDecisionOutput], self)
        chain = runnable | RunnableLambda(lambda result: result.model_dump(mode="json", exclude_none=True))
        return chain.as_tool(
            args_schema=args_schema,
            name=name or "kairos_decide",
            arg_types=arg_types,
            description=description or "Ask a configured Kairos action for a structured risk decision.",
        )

    def to_json(self):
        # Reconstructing this Runnable without its caller-owned client changes its behavior.
        from langchain_core.load.serializable import to_json_not_implemented

        return to_json_not_implemented(self)


class KairosToolkit(BaseToolkit):
    """Explicit Kairos tools for LangChain agents; the client remains caller-owned.

    ``decide`` is the default operation. Select event tracking and workflow execution
    explicitly because they write to Kairos. ``track`` only queues an event in the
    SDK's background sender; use ``flush`` when the application needs to wait for the
    queue to drain before proceeding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)

    client: Any = Field(exclude=True, repr=False)
    operations: tuple[KairosOperation, ...] = ("decide",)
    _kairos: KairosClientProtocol = PrivateAttr()

    @model_validator(mode="after")
    def validate_operations(self) -> Self:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("Select each Kairos operation at most once")
        self._kairos = self.client
        return self

    def _call(self, operation: KairosOperation, request: BaseModel) -> Any:
        data = request.model_dump(exclude_none=True)
        if operation == "decide":
            result = self._kairos.decide(data.pop("action"), data.pop("properties"), **data)
            return _decision_output(result).model_dump(mode="json", exclude_none=True)
        if operation == "track":
            self._kairos.track(data.pop("event_type"), data.pop("properties"), **data)
            return {"queued": True, "delivery": "background"}
        if operation == "track_and_decide":
            result = self._kairos.track_and_decide(data.pop("event_type"), data.pop("properties"), **data)
            return _decision_output(result).model_dump(mode="json", exclude_none=True)
        if operation == "run_workflow":
            return self._kairos.run_workflow(data.pop("name"), data.pop("input"), **data)
        if operation == "flush":
            self._kairos.flush()
            return {"flushed": True}
        raise ValueError(f"Unsupported Kairos operation: {operation}")

    async def _acall(self, operation: KairosOperation, request: BaseModel) -> Any:
        return await asyncio.to_thread(self._call, operation, request)

    def _tool(self, operation: KairosOperation) -> BaseTool:
        schema: Any = {
            "decide": KairosDecisionInput,
            "track": KairosEventInput,
            "track_and_decide": KairosTrackAndDecideInput,
            "run_workflow": KairosWorkflowInput,
            "flush": _FlushInput,
        }[operation]
        descriptions = {
            "decide": (
                "Ask a configured Kairos action for approve, step_up, review, or block. "
                "Include the relevant entity IDs."
            ),
            "track": (
                "Queue an event in Kairos history. Delivery happens in the background; "
                "select flush to wait for queued events."
            ),
            "track_and_decide": (
                "Record the triggering event and ask its configured Kairos action for a decision in one call."
            ),
            "run_workflow": (
                "Start a named Kairos workflow. This can trigger remote workflow actions; select explicitly."
            ),
            "flush": "Wait for the caller-owned Kairos client's queued events to be sent.",
        }

        def invoke(**kwargs: Any) -> Any:
            return self._call(operation, schema.model_validate(kwargs))

        async def ainvoke(**kwargs: Any) -> Any:
            return await self._acall(operation, schema.model_validate(kwargs))

        return StructuredTool.from_function(
            func=invoke,
            coroutine=ainvoke,
            name=f"kairos_{operation}",
            description=descriptions[operation],
            args_schema=schema,
        )

    def get_tools(self) -> list[BaseTool]:
        return [self._tool(operation) for operation in self.operations]


class _FlushInput(BaseModel):
    """No-argument schema for the explicit event queue flush operation."""

    model_config = ConfigDict(extra="forbid")
