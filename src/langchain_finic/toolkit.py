"""Explicitly selected Finic lifecycle tools with a caller-owned client."""

from collections.abc import Callable
from typing import Any, Literal, Self

from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_finic.client import FinicClient
from langchain_finic.models import Model, Text
from langchain_finic.scoring_models import (
    AddQuestionsRequest,
    CalibrationRequest,
    CreateScoringModelRequest,
    Criteria,
    ModelName,
    QuestionId,
    Questions,
    Rows,
    UpdateScoringModelRequest,
)

Operation = Literal[
    "list_scoring_models",
    "get_scoring_model",
    "get_job_status",
    "create_scoring_model",
    "add_questions",
    "remove_question",
    "update_scoring_model",
    "calibrate",
    "delete_scoring_model",
]

READ_OPERATIONS: tuple[Operation, ...] = ("list_scoring_models", "get_scoring_model", "get_job_status")
# Deletion removes a scoring model and its records with no way back, so it is
# never a default: an application selects it deliberately or not at all.
DESTRUCTIVE_OPERATIONS: frozenset[str] = frozenset({"delete_scoring_model"})


class _NoArguments(Model):
    pass


class _JobArguments(Model):
    job_id: Text


class _ModelArguments(Model):
    model_id: Text


class _QuestionArguments(Model):
    model_id: Text
    question_id: QuestionId


class _AddQuestionsArguments(Model):
    model_id: Text
    questions: Questions


class _UpdateArguments(Model):
    model_id: Text
    name: ModelName | None = None
    criteria: Criteria | None = None
    archived: bool | None = None


class _CreateArguments(Model):
    name: ModelName
    criteria: Criteria | None = None
    data: Rows | None = None
    add_questions: Questions | None = None


class _CalibrateArguments(Model):
    model_id: Text
    data: Rows
    add_questions: Questions | None = None
    remove_questions: list[QuestionId] | None = None


def _fields(arguments: Model, *drop: str) -> dict[str, Any]:
    return {key: value for key, value in arguments.model_dump(exclude_none=True).items() if key not in drop}


def _no_args(arguments: Model) -> tuple[Any, ...]:
    return ()


def _job(arguments: Model) -> tuple[Any, ...]:
    assert isinstance(arguments, _JobArguments)
    return (arguments.job_id,)


def _model(arguments: Model) -> tuple[Any, ...]:
    assert isinstance(arguments, (_ModelArguments, _QuestionArguments))
    if isinstance(arguments, _QuestionArguments):
        return (arguments.model_id, arguments.question_id)
    return (arguments.model_id,)


def _create(arguments: Model) -> tuple[Any, ...]:
    return (CreateScoringModelRequest.model_validate(_fields(arguments)),)


def _calibrate(arguments: Model) -> tuple[Any, ...]:
    return (CalibrationRequest.model_validate(_fields(arguments)),)


def _add_questions(arguments: Model) -> tuple[Any, ...]:
    assert isinstance(arguments, _AddQuestionsArguments)
    return (arguments.model_id, AddQuestionsRequest.model_validate(_fields(arguments, "model_id")))


def _update(arguments: Model) -> tuple[Any, ...]:
    assert isinstance(arguments, _UpdateArguments)
    return (
        arguments.model_id,
        UpdateScoringModelRequest.model_validate(_fields(arguments, "model_id")),
    )


_SCHEMAS: dict[Operation, type[Model]] = {
    "list_scoring_models": _NoArguments,
    "get_scoring_model": _ModelArguments,
    "get_job_status": _JobArguments,
    "create_scoring_model": _CreateArguments,
    "add_questions": _AddQuestionsArguments,
    "remove_question": _QuestionArguments,
    "update_scoring_model": _UpdateArguments,
    "calibrate": _CalibrateArguments,
    "delete_scoring_model": _ModelArguments,
}

_ARGUMENTS: dict[Operation, Callable[[Model], tuple[Any, ...]]] = {
    "list_scoring_models": _no_args,
    "get_scoring_model": _model,
    "get_job_status": _job,
    "create_scoring_model": _create,
    "add_questions": _add_questions,
    "remove_question": _model,
    "update_scoring_model": _update,
    "calibrate": _calibrate,
    "delete_scoring_model": _model,
}

_DESCRIPTIONS: dict[Operation, str] = {
    "list_scoring_models": (
        "List up to 500 Finic scoring models, newest first, with readiness, criteria and counts."
    ),
    "get_scoring_model": (
        "Read one scoring model with the questions it uses and their weights, strongest first."
    ),
    "get_job_status": "Check a job once. Complete may mean failure: inspect error and retain job_id.",
    "create_scoring_model": (
        "Create a scoring model from criteria alone, or from at least 100 labeled records. "
        "Returns model_id, and job_id only when records are being fitted. Do not blindly retry."
    ),
    "add_questions": (
        "Add up to 20 of your own yes/no questions to a scoring model without calibrating it. "
        "Adding a question it already has is rejected, so do not blindly retry."
    ),
    "remove_question": (
        "Remove one question from a scoring model. A question its calibrated score uses is "
        "refused; remove that one through calibrate instead."
    ),
    "update_scoring_model": (
        "Rename a scoring model, change its criteria, or archive and unarchive it. "
        "Archiving stops scoring and calibration without deleting anything."
    ),
    "calibrate": (
        "Fit a scoring model to at least 100 labeled records, optionally adding and removing "
        "questions. Returns a job without waiting. Do not blindly retry."
    ),
    "delete_scoring_model": (
        "Permanently delete a scoring model and its records. This cannot be undone and needs an "
        "admin's API key. Archive it with update_scoring_model instead unless deletion is intended."
    ),
}


class FinicToolkit(BaseToolkit):
    """Borrow a client; expose read operations by default and writes only by selection.

    Keep the supplied client open while tools run. Creation/calibration return
    accepted job handles immediately; polling is an explicit application operation.
    Deletion is irreversible and is never selected by default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)
    client: FinicClient = Field(exclude=True, repr=False)
    operations: tuple[Operation, ...] = READ_OPERATIONS

    @model_validator(mode="after")
    def unique_operations(self) -> Self:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("Select each operation at most once")
        return self

    @property
    def destructive(self) -> tuple[Operation, ...]:
        """The selected operations that cannot be undone."""
        return tuple(name for name in self.operations if name in DESTRUCTIVE_OPERATIONS)

    def _tool(self, operation: Operation) -> BaseTool:
        schema = _SCHEMAS[operation]
        build = _ARGUMENTS[operation]

        def arguments(kwargs: dict[str, Any]) -> tuple[Any, ...]:
            return build(schema.model_validate(kwargs))

        def invoke(**kwargs: Any) -> dict[str, Any]:
            result = getattr(self.client, operation)(*arguments(kwargs))
            return self._dump(result)

        async def ainvoke(**kwargs: Any) -> dict[str, Any]:
            result = await getattr(self.client, "a" + operation)(*arguments(kwargs))
            return self._dump(result)

        return StructuredTool.from_function(
            func=invoke,
            coroutine=ainvoke,
            name="finic_" + operation,
            description=_DESCRIPTIONS[operation],
            args_schema=schema,
        )

    @staticmethod
    def _dump(result: BaseModel) -> dict[str, Any]:
        return result.model_dump(mode="json", exclude_none=True)

    def get_tools(self) -> list[BaseTool]:
        return [self._tool(operation) for operation in self.operations]
