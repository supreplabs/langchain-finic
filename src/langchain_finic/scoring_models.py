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


class CalibrationRequest(Model):
    model_id: Text
    data: Rows

    @field_validator("data", mode="before")
    @classmethod
    def json_values(cls, value):
        return normalize_state(value)

    @model_validator(mode="after")
    def outcomes(self) -> Self:
        rows = _rows(self.data)
        if not rows:
            raise ValueError("Calibration requires outcomes")
        for row in rows:
            score_id = row.get("score_id")
            if not isinstance(score_id, str) or not score_id.strip() or len(score_id) > 200:
                raise ValueError("Each outcome needs a score_id of 1 to 200 characters")
            _label(row.get("is_bad"))
            if "model_id" in row and row["model_id"] != self.model_id:
                raise ValueError("Outcome model_id must match the request model_id")
        return self


class ResponseModel(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)


class ScoringResponse(ResponseModel):
    model_id: Text
    score_id: ScoreId
    score: Annotated[StrictInt, Field(ge=0, le=1000)]
    request_id: Text

    def validate_for(self, request: ScoringRequest) -> None:
        if self.model_id != request.model_id or self.score_id != request.score_id:
            raise ValueError("Scoring response identity does not match the request")


class CreateScoreResponse(ResponseModel):
    model_id: Text
    job_id: Text
    request_id: Text


class CalibrationResponse(ResponseModel):
    job_id: Text
    matched: Annotated[StrictInt, Field(ge=0)]
    unmatched: Annotated[StrictInt, Field(ge=0)]
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
