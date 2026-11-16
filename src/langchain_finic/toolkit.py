"""Explicitly selected Finic lifecycle tools with a caller-owned client."""

from typing import Annotated, Any, Literal, Self, cast

from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool
from pydantic import ConfigDict, Field, StrictBool, create_model, field_validator, model_validator

from langchain_finic.client import FinicClient
from langchain_finic.models import DecisionRequest, Model, Text
from langchain_finic.scoring_models import (
    AddQuestionsRequest,
    CalibrationRequest,
    CreateScoreRequest,
    CreateScoringModelRequest,
    ScoringRequest,
    UpdateScoringModelRequest,
)

Operation = Literal[
    "ask",
    "score",
    "list_scores",
    "list_scoring_models",
    "get_job_status",
    "get_scoring_model",
    "create_score",
    "create_scoring_model",
    "update_scoring_model",
    "add_questions",
    "remove_question",
    "delete_scoring_model",
    "calibrate",
]


class _NoArguments(Model):
    pass


class _JobArguments(Model):
    job_id: Text


class _ModelArguments(Model):
    model_id: Text


class _RemoveQuestionArguments(_ModelArguments):
    question_id: Text


class _UpdateModelArguments(_ModelArguments):
    name: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    criteria: Annotated[str, Field(max_length=4000)] | None = None
    archived: StrictBool | None = None

    def request(self) -> UpdateScoringModelRequest:
        return UpdateScoringModelRequest.model_validate(
            {
                key: value
                for key, value in self.model_dump(exclude={"model_id"}).items()
                if key in self.model_fields_set and value is not None
            }
        )


class FinicToolkit(BaseToolkit):
    """Borrow a client; expose read operations by default and writes only by selection.

    Keep the supplied client open while tools run. Creation/calibration return
    accepted job handles immediately; polling is an explicit application operation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)
    client: FinicClient = Field(exclude=True, repr=False)
    allow_schema_update: StrictBool = False
    allow_schema_bypass: StrictBool = False
    operations: tuple[Operation, ...] = ("list_scoring_models", "get_job_status")

    @model_validator(mode="after")
    def unique_operations(self) -> Self:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("Select each operation at most once")
        return self

    def _tool(self, operation: Operation) -> BaseTool:
        schemas: dict[str, type[Model]] = {
            "ask": DecisionRequest,
            "score": ScoringRequest,
            "list_scores": _NoArguments,
            "list_scoring_models": _NoArguments,
            "get_job_status": _JobArguments,
            "get_scoring_model": _ModelArguments,
            "create_score": CreateScoreRequest,
            "create_scoring_model": CreateScoringModelRequest,
            "update_scoring_model": _UpdateModelArguments,
            "add_questions": _AddQuestionsArguments,
            "remove_question": _RemoveQuestionArguments,
            "delete_scoring_model": _ModelArguments,
            "calibrate": CalibrationRequest,
        }
        descriptions = {
            "ask": (
                "Ask Grayson a typed question about a case. The response contains probabilities, "
                "not instructions to take an action."
            ),
            "score": (
                "Score one record from 0 (lowest risk) to 1000 (highest). Supply a unique score_id; "
                "never reuse one after a timeout."
            ),
            "list_scores": "List up to 500 Finic models, newest first, with readiness and outcome counts.",
            "list_scoring_models": (
                "List up to 500 scoring models, newest first, including status and calibration summary."
            ),
            "get_job_status": "Check a job once. Complete may mean failure: inspect error and retain job_id.",
            "get_scoring_model": (
                "Get one scoring model and its questions, weights, status, and calibration details."
            ),
            "create_score": (
                "Create a persistent model from labeled examples. "
                "Returns model_id/job_id without waiting. Do not blindly retry."
            ),
            "create_scoring_model": (
                "Create a persistent scoring model from labeled records or decision criteria. "
                "Returns a model_id and sometimes a job_id. Do not blindly retry."
            ),
            "update_scoring_model": (
                "Rename, change criteria, archive, or unarchive a scoring model. Archiving is reversible."
            ),
            "add_questions": (
                "Add up to 20 yes/no questions to a scoring model. Do not blindly retry after a timeout."
            ),
            "remove_question": (
                "Remove an unused question by id. Questions in a calibrated score must be removed "
                "via calibration."
            ),
            "delete_scoring_model": (
                "Permanently delete a scoring model and its records. Prefer reversible archive "
                "unless deletion is explicitly intended."
            ),
            "calibrate": (
                "Fit a scoring model using at least 100 labeled records. "
                "Returns a job without waiting. Do not blindly retry."
            ),
        }
        schema = schemas[operation]
        if operation == "score":
            disabled = {
                key
                for key, enabled in (
                    ("update_schema", self.allow_schema_update),
                    ("skip_schema_validation", self.allow_schema_bypass),
                )
                if not enabled
            }
            fields: dict[str, Any] = {
                key: (field.annotation or Any, field)
                for key, field in ScoringRequest.model_fields.items()
                if key not in disabled
            }
            schema = create_model(
                "FinicToolkitScoringInput",
                __base__=Model,
                **fields,
            )

        def arguments(kwargs):
            if operation == "score":
                for flag, enabled in (
                    ("update_schema", self.allow_schema_update),
                    ("skip_schema_validation", self.allow_schema_bypass),
                ):
                    if flag in kwargs and kwargs[flag] is not None and not enabled:
                        raise ValueError("Scoring schema changes/bypass require explicit toolkit opt-in")
            validated = schema.model_validate(kwargs)
            if operation in {"list_scores", "list_scoring_models"}:
                return ()
            if operation == "get_job_status":
                return (cast(_JobArguments, validated).job_id,)
            if operation == "get_scoring_model" or operation == "delete_scoring_model":
                return (cast(_ModelArguments, validated).model_id,)
            if operation == "update_scoring_model":
                update = cast(_UpdateModelArguments, validated)
                return (update.model_id, update.request())
            if operation == "add_questions":
                addition = cast(_AddQuestionsArguments, validated)
                return (addition.model_id, AddQuestionsRequest(questions=addition.questions))
            if operation == "remove_question":
                removal = cast(_RemoveQuestionArguments, validated)
                return (removal.model_id, removal.question_id)
            if operation == "score":
                return (ScoringRequest.model_validate(validated.model_dump(exclude_none=True)),)
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


class _AddQuestionsArguments(_ModelArguments):
    questions: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(min_length=1, max_length=20)

    @field_validator("questions")
    @classmethod
    def nonblank_questions(cls, value):
        if any(not question.strip() for question in value):
            raise ValueError("Questions must not be blank")
        return value
