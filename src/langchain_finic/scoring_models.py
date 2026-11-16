"""Typed contracts for custom scoring and asynchronous model lifecycle jobs."""

import csv
import io
import json
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from langchain_finic._state import normalize_state
from langchain_finic.models import Model, Probability, Text

ScoreId = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
Record = dict[str, JsonValue]
Rows = list[Record] | Text


def _rows(data: Rows | Record) -> list[Record]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    try:
        reader = csv.reader(io.StringIO(data), strict=True)
        header = next(reader)
        if not header or any(not key.strip() for key in header) or len(set(header)) != len(header):
            raise ValueError("CSV must have unique nonblank headers")
        rows: list[Record] = []
        for values in reader:
            if len(values) != len(header):
                raise ValueError("CSV rows must match the header width")
            rows.append(dict(zip(header, values, strict=True)))
        return rows
    except (csv.Error, StopIteration):
        raise ValueError("Invalid CSV data") from None


def _label(value: JsonValue) -> bool:
    if isinstance(value, bool) or type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "0", "1", "yes", "no"}:
        return value.lower() in {"true", "1", "yes"}
    raise ValueError("Each outcome must be a boolean, integer zero/one, or text true/false/1/0/yes/no")


def _current_label(value: JsonValue) -> bool:
    if isinstance(value, bool) or type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false", "0", "1"}:
        return value.lower() in {"true", "1"}
    raise ValueError("Each outcome must be true or false (or integer zero/one)")


def _check_record(row: Record) -> None:
    features = {key: value for key, value in row.items() if key != "is_bad"}
    if not features:
        raise ValueError("Each record needs at least one field other than is_bad")
    if (
        len(json.dumps(features, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
        > 60_000
    ):
        raise ValueError("Record exceeds the 60,000 byte compact JSON limit")


class ScoringInput(Model):
    """Input to a scorer bound to a custom model; IDs are supplied by the caller."""

    score_id: ScoreId
    data: Record | Rows
    skip_schema_validation: StrictBool | None = Field(
        default=None, description="Bypass the stored model schema validation for this record."
    )
    update_schema: StrictBool | None = Field(
        default=None, description="Change the remote model schema to this record for future scoring calls."
    )
    include_questions: StrictBool | None = Field(
        default=None, description="Include each question's probability and weight in the score response."
    )

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def single_record(self) -> Self:
        rows = _rows(self.data)
        if len(rows) != 1:
            raise ValueError("Scoring requires exactly one record")
        # CSV coercion/dotted-field expansion and its resulting byte size belong to Finic.
        if not isinstance(self.data, str):
            _check_record(rows[0])
        elif not set(rows[0]) - {"is_bad"}:
            raise ValueError("Each record needs at least one field other than is_bad")
        return self


class ScoringRequest(ScoringInput):
    model_id: Text


class CreateScoreRequest(Model):
    data: Rows
    description: Annotated[str, Field(max_length=4000)] | None = None

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def training_examples(self) -> Self:
        rows = _rows(self.data)
        if len(rows) < 100:
            raise ValueError("Creating a score requires at least 100 labeled rows")
        labels = []
        for row in rows:
            labels.append(_label(row.get("is_bad")))
            if not isinstance(self.data, str):
                _check_record(row)
            elif not set(row) - {"is_bad"}:
                raise ValueError("Each example needs at least one field other than is_bad")
        if sum(labels) < 20 or len(labels) - sum(labels) < 20:
            raise ValueError("Creating a score requires at least 20 examples of each outcome")
        return self


class ResponseModel(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)


class ScoringResponse(ResponseModel):
    model_id: Text
    score_id: ScoreId
    score: Annotated[StrictInt, Field(ge=0, le=1000)]
    request_id: Text
    questions: list["ScoringQuestionResult"] | None = None

    def validate_for(self, request: ScoringRequest) -> None:
        if self.model_id != request.model_id or self.score_id != request.score_id:
            raise ValueError("Scoring response identity does not match the request")


class CreateScoreResponse(ResponseModel):
    model_id: Text
    job_id: Text
    request_id: Text


class ScoreModel(ResponseModel):
    model_id: Text
    status: Literal["training", "ready", "failed"]
    criteria: str
    created_at: Text
    calibrated_at: str | None
    examples: Annotated[StrictInt, Field(ge=0)]
    bad_rate: Probability
    scored: Annotated[StrictInt, Field(ge=0)]
    outcomes: Annotated[StrictInt, Field(ge=0)]


class ScoringQuestion(ResponseModel):
    """Question and contribution metadata returned for a scoring model."""

    id: Text
    text: Text
    source: Literal["finic", "user"]
    weight: Annotated[float, Field(allow_inf_nan=False)]


class ScoringQuestionResult(ResponseModel):
    """Per-record question evidence optionally included with a score."""

    id: Text
    text: Text
    probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    weight: Annotated[float, Field(allow_inf_nan=False)]


class ScoringModel(ResponseModel):
    """Current Finic scoring-model representation."""

    model_id: Text
    name: str
    status: Literal["training", "ready", "failed"]
    criteria: str
    created_at: Text
    calibrated_at: str | None
    archived_at: str | None
    examples: Annotated[StrictInt, Field(ge=0)]
    bad_rate: Probability
    scored: Annotated[StrictInt, Field(ge=0)]
    questions: list[ScoringQuestion] | None = None
    added: list["AddedQuestion"] | None = None


class AddedQuestion(ResponseModel):
    id: Text
    text: Text
    used: StrictBool


def _validate_labeled_rows(data: Rows) -> None:
    rows = _rows(data)
    if len(rows) < 100:
        raise ValueError("Finic requires at least 100 labeled rows")
    labels: list[bool] = []
    for row in rows:
        labels.append(_current_label(row.get("is_bad")))
        if not isinstance(data, str):
            _check_record(row)
        elif not set(row) - {"is_bad"}:
            raise ValueError("Each example needs at least one field other than is_bad")
    if sum(labels) < 20 or len(labels) - sum(labels) < 20:
        raise ValueError("Finic requires at least 20 examples of each outcome")


class CreateScoringModelRequest(Model):
    """Create a model from labeled records or from a name and decision criteria."""

    name: Annotated[str, Field(min_length=1, max_length=100)]
    criteria: Annotated[str, Field(max_length=4000)] | None = None
    data: Rows | None = None
    add_questions: list[Annotated[str, Field(min_length=1, max_length=500)]] | None = Field(
        default=None, max_length=20
    )

    @field_validator("name", "criteria")
    @classmethod
    def nonblank_when_present(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Value must not be blank")
        return value

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def valid_creation_mode(self) -> Self:
        if self.data is None:
            if self.criteria is None:
                raise ValueError("Provide labeled data or criteria")
        else:
            _validate_labeled_rows(self.data)
        if self.add_questions is not None and any(not question.strip() for question in self.add_questions):
            raise ValueError("Questions must not be blank")
        return self


class CalibrationRequest(Model):
    model_id: Text
    data: Rows
    add_questions: list[Annotated[str, Field(min_length=1, max_length=500)]] | None = Field(
        default=None, max_length=20
    )
    remove_questions: list[Text] | None = Field(default=None, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def earlier_examples_name(cls, value):
        if isinstance(value, dict) and "examples" in value:
            if "data" in value:
                raise ValueError("Pass data or earlier examples, not both")
            value = {**value, "data": value["examples"]}
            value.pop("examples", None)
        return value

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def outcomes(self) -> Self:
        _validate_labeled_rows(self.data)
        if self.add_questions is not None and any(not question.strip() for question in self.add_questions):
            raise ValueError("Questions must not be blank")
        if self.remove_questions is not None and any(
            not question.strip() for question in self.remove_questions
        ):
            raise ValueError("Question IDs must not be blank")
        return self


class UpdateScoringModelRequest(Model):
    name: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    criteria: Annotated[str, Field(max_length=4000)] | None = None
    archived: StrictBool | None = None

    @model_validator(mode="after")
    def at_least_one_change(self) -> Self:
        if not any(value is not None for value in (self.name, self.criteria, self.archived)):
            raise ValueError("Provide name, criteria, or archived")
        if self.name is not None and not self.name.strip():
            raise ValueError("name must not be blank")
        return self


class AddQuestionsRequest(Model):
    questions: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(min_length=1, max_length=20)

    @field_validator("questions")
    @classmethod
    def nonblank_questions(cls, value):
        if any(not question.strip() for question in value):
            raise ValueError("Questions must not be blank")
        return value


class DeleteScoringModelResponse(ResponseModel):
    model_id: Text
    deleted: StrictBool

    @model_validator(mode="after")
    def actually_deleted(self) -> Self:
        if not self.deleted:
            raise ValueError("Finic did not confirm model deletion")
        return self


class ListScoringModelsResponse(ResponseModel):
    scoring_models: list[ScoringModel]


class CreateScoringModelResponse(ResponseModel):
    """Response for either immediate or asynchronous model creation."""

    model_id: Text
    request_id: Text
    job_id: Text | None = None


class CalibrationResponse(ResponseModel):
    job_id: Text
    request_id: Text
    rows: Annotated[StrictInt, Field(ge=0)] | None = None
    # Kept for responses from the earlier calibration contract.
    matched: Annotated[StrictInt, Field(ge=0)] | None = None
    unmatched: Annotated[StrictInt, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def has_row_count(self) -> Self:
        if self.rows is None and (self.matched is None or self.unmatched is None):
            raise ValueError("Calibration response must include rows or matched and unmatched counts")
        return self


class ListScoresResponse(ResponseModel):
    scores: list[ScoreModel]


class JobFailure(ResponseModel):
    type: Text
    message: str = Field(repr=False)


class JobStatus(ResponseModel):
    job_id: Text
    complete: StrictBool
    error: JobFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.complete and self.error is None

    @model_validator(mode="after")
    def failure_is_complete(self) -> Self:
        if self.error is not None and not self.complete:
            raise ValueError("A failed job must be complete")
        return self
