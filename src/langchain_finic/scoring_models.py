"""Typed contracts for custom scoring and asynchronous scoring-model lifecycle jobs."""

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
QuestionId = Annotated[str, Field(min_length=1, max_length=200, pattern=r"\S")]
# Finic accepts up to 20 questions per call and 500 characters each.
QuestionText = Annotated[str, Field(min_length=1, max_length=500, pattern=r"\S")]
Questions = Annotated[list[QuestionText], Field(min_length=1, max_length=20)]
ModelName = Annotated[str, Field(min_length=1, max_length=100, pattern=r"\S")]
Criteria = Annotated[str, Field(max_length=4000)]
Weight = Annotated[float, Field(allow_inf_nan=False)]
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


def _check_record(row: Record) -> None:
    features = {key: value for key, value in row.items() if key != "is_bad"}
    if not features:
        raise ValueError("Each record needs at least one field other than is_bad")
    if (
        len(json.dumps(features, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
        > 60_000
    ):
        raise ValueError("Record exceeds the 60,000 byte compact JSON limit")


def _check_labeled_rows(data: Rows) -> None:
    """Apply Finic's labeled-record minima, shared by creation and calibration."""
    rows = _rows(data)
    if len(rows) < 100:
        raise ValueError("Labeled records require at least 100 rows")
    labels = []
    for row in rows:
        labels.append(_label(row.get("is_bad")))
        if not isinstance(data, str):
            _check_record(row)
        elif not set(row) - {"is_bad"}:
            raise ValueError("Each labeled record needs at least one field other than is_bad")
    if sum(labels) < 20 or len(labels) - sum(labels) < 20:
        raise ValueError("Labeled records require at least 20 examples of each outcome")


class ScoringInput(Model):
    """Input to a scorer bound to a scoring model; IDs are supplied by the caller."""

    score_id: ScoreId
    data: Record | Rows
    skip_schema_validation: StrictBool | None = Field(
        default=None, description="Bypass the stored model schema validation for this record."
    )
    update_schema: StrictBool | None = Field(
        default=None, description="Change the remote model schema to this record for future scoring calls."
    )
    include_questions: StrictBool | None = Field(
        default=None,
        description="Also return the questions the score used, each with its probability and weight.",
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


class CreateScoringModelRequest(Model):
    """Create a scoring model from its criteria alone, or from labeled records."""

    name: ModelName
    criteria: Criteria | None = None
    data: Rows | None = None
    add_questions: Questions | None = None

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def criteria_or_records(self) -> Self:
        if self.data is None:
            if self.criteria is None or not self.criteria.strip():
                raise ValueError("Provide criteria, labeled records in data, or both")
            return self
        _check_labeled_rows(self.data)
        return self


class UpdateScoringModelRequest(Model):
    """Rename a scoring model, change its criteria, or archive and unarchive it."""

    name: ModelName | None = None
    # Finic allows empty criteria only once the scoring model is calibrated.
    criteria: Criteria | None = None
    archived: StrictBool | None = None

    @model_validator(mode="after")
    def at_least_one_change(self) -> Self:
        if self.name is None and self.criteria is None and self.archived is None:
            raise ValueError("Provide name, criteria, or archived")
        return self


class AddQuestionsRequest(Model):
    """Add your own yes/no questions to a scoring model without calibrating it."""

    questions: Questions


class CalibrationRequest(Model):
    """Fit a scoring model to labeled records, optionally changing its questions."""

    model_id: Text
    data: Rows
    add_questions: Questions | None = None
    remove_questions: Annotated[list[QuestionId], Field(min_length=1, max_length=50)] | None = None

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def labeled_records(self) -> Self:
        _check_labeled_rows(self.data)
        if self.remove_questions is not None and len(set(self.remove_questions)) != len(
            self.remove_questions
        ):
            raise ValueError("Each removed question id must appear once")
        return self


class ResponseModel(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)


class ScoredQuestion(ResponseModel):
    """A question behind one score, returned with include_questions."""

    id: Text
    text: str
    probability: Probability
    weight: Weight


class ScoringResponse(ResponseModel):
    model_id: Text
    score_id: ScoreId
    score: Annotated[StrictInt, Field(ge=0, le=1000)]
    questions: list[ScoredQuestion] | None = None
    request_id: Text

    def validate_for(self, request: ScoringRequest) -> None:
        if self.model_id != request.model_id or self.score_id != request.score_id:
            raise ValueError("Scoring response identity does not match the request")
        if not request.include_questions and self.questions is not None:
            raise ValueError("Finic returned questions that were not requested")


class CreateScoringModelResponse(ResponseModel):
    """A created scoring model; job_id is present only when records are being fitted."""

    model_id: Text
    job_id: Text | None = None
    request_id: Text

    @property
    def ready(self) -> bool:
        """True when the scoring model was created from criteria alone and needs no job."""
        return self.job_id is None


class CalibrationResponse(ResponseModel):
    job_id: Text
    rows: Annotated[StrictInt, Field(ge=0)]
    request_id: Text


class ScoringModelQuestion(ResponseModel):
    """A question a scoring model uses, with its calibrated or estimated weight."""

    id: Text
    text: str
    source: Literal["finic", "user"]
    weight: Weight


class ScoringModel(ResponseModel):
    """A scoring model as listing returns it. Name and criteria may be empty."""

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

    @property
    def archived(self) -> bool:
        return self.archived_at is not None

    @property
    def calibrated(self) -> bool:
        return self.calibrated_at is not None


class ScoringModelDetail(ScoringModel):
    """One scoring model with the questions it uses, strongest first."""

    questions: list[ScoringModelQuestion]

    @property
    def user_questions(self) -> list[ScoringModelQuestion]:
        return [question for question in self.questions if question.source == "user"]


class AddedQuestion(ResponseModel):
    id: Text
    text: str
    used: StrictBool


class AddQuestionsResponse(ScoringModelDetail):
    added: list[AddedQuestion]


class DeleteScoringModelResponse(ResponseModel):
    model_id: Text
    deleted: Literal[True]


class ListScoringModelsResponse(ResponseModel):
    scoring_models: list[ScoringModel]


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
