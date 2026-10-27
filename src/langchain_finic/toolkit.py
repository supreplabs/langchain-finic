"""Explicitly selected Finic lifecycle tools with a caller-owned client."""

from typing import Literal, Self

from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool
from pydantic import ConfigDict, Field, model_validator

from langchain_finic.client import FinicClient
from langchain_finic.models import Model, Text
from langchain_finic.scoring_models import CalibrationRequest, CreateScoreRequest

Operation = Literal["list_scores", "get_job_status", "create_score", "calibrate"]


class _NoArguments(Model):
    pass


class _JobArguments(Model):
    job_id: Text


class FinicToolkit(BaseToolkit):
    """Borrow a client; expose read operations by default and writes only by selection.

    Keep the supplied client open while tools run. Creation/calibration return
    accepted job handles immediately; polling is an explicit application operation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)
    client: FinicClient = Field(exclude=True, repr=False)
    operations: tuple[Operation, ...] = ("list_scores", "get_job_status")

    @model_validator(mode="after")
    def unique_operations(self) -> Self:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("Select each operation at most once")
        return self

    def _tool(self, operation: Operation) -> BaseTool:
        schemas: dict[str, type[Model]] = {
            "list_scores": _NoArguments,
            "get_job_status": _JobArguments,
            "create_score": CreateScoreRequest,
            "calibrate": CalibrationRequest,
        }
        descriptions = {
            "list_scores": "List up to 500 Finic models, newest first, with readiness and outcome counts.",
            "get_job_status": "Check a job once. Complete may mean failure: inspect error and retain job_id.",
            "create_score": (
                "Create a persistent model from labeled examples. "
                "Returns model_id/job_id without waiting. Do not blindly retry."
            ),
            "calibrate": (
                "Update a model using outcomes for existing score IDs. "
                "Returns a job without waiting. Do not blindly retry."
            ),
        }
        schema = schemas[operation]

        def arguments(kwargs):
            validated = schema.model_validate(kwargs)
            if operation == "list_scores":
                return ()
            if operation == "get_job_status":
                return (_JobArguments.model_validate(kwargs).job_id,)
            return (validated,)

        def invoke(**kwargs):
            result = getattr(self.client, operation)(*arguments(kwargs))
            return result.model_dump(mode="json", exclude_none=True)

        async def ainvoke(**kwargs):
            result = await getattr(self.client, "a" + operation)(*arguments(kwargs))
            return result.model_dump(mode="json", exclude_none=True)

        return StructuredTool.from_function(
            func=invoke,
            coroutine=ainvoke,
            name="finic_" + operation,
            description=descriptions[operation],
            args_schema=schema,
        )

    def get_tools(self) -> list[BaseTool]:
        return [self._tool(operation) for operation in self.operations]
