"""Pydantic models for Finic Grayson's decision API."""

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from langchain_finic._config import env_option
from langchain_finic._state import normalize_state

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
ModelName = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]
Name = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class NoulQuestion(Model):
    type: Literal["noul"] = "noul"
    instructions: Text
    criteria: dict[Literal["true", "false"], Text] | None = None
    base_rate: Annotated[float, Field(gt=0, lt=1, allow_inf_nan=False)] | None = None


class ChoiceQuestion(Model):
    type: Literal["choice", "multi_choice"]
    instructions: Text
    criteria: Annotated[dict[Name, Text | None], Field(min_length=2, max_length=255)]
    base_rate: dict[Name, Probability] | None = None

    @model_validator(mode="after")
    def unique_criteria(self) -> Self:
        values = [
            (value if value is not None else key.replace("_", " ")).strip().casefold()
            for key, value in self.criteria.items()
        ]
        if len(set(values)) != len(values):
            raise ValueError("Criteria descriptions must be unique")
        if self.base_rate is not None:
            if not self.base_rate.keys() <= self.criteria.keys():
                raise ValueError("Base-rate keys must be criteria keys")
            if self.type == "choice" and (
                self.base_rate.keys() != self.criteria.keys()
                or not math.isclose(sum(self.base_rate.values()), 1, abs_tol=0.020000000001)
            ):
                raise ValueError("Choice base rates must cover every option and sum to one")
            if self.type == "choice":
                total = sum(self.base_rate.values())
                self.base_rate = {key: rate / total for key, rate in self.base_rate.items()}
            elif any(not 0 < rate < 1 for rate in self.base_rate.values()):
                raise ValueError("Multi-choice base rates must be strictly between zero and one")
        return self


class ScoreQuestion(Model):
    type: Literal["score"] = "score"
    instructions: Text
    criteria: Annotated[list[Text], Field(min_length=2, max_length=32)]
    base_rate: list[Probability] | None = None

    @model_validator(mode="after")
    def unique_criteria(self) -> Self:
        values = [value.strip().casefold() for value in self.criteria]
        if len(set(values)) != len(values):
            raise ValueError("Score levels must be unique")
        if self.base_rate is not None and (
            len(self.base_rate) != len(self.criteria)
            or not math.isclose(sum(self.base_rate), 1, abs_tol=0.020000000001)
        ):
            raise ValueError("Score base rates must cover every level and sum to one")
        return self


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionRequest(Model):
    model: ModelName = Field(default_factory=lambda: env_option("MODEL", "grayson-1"), validate_default=True)
    data: Text | dict[str, JsonValue] | list[JsonValue] | None = None
    context: Text | None = None
    questions: Annotated[dict[Name, Question], Field(min_length=1, max_length=128)]

    @model_validator(mode="before")
    @classmethod
    def legacy_state(cls, value):
        if isinstance(value, dict) and "state" in value:
            if "data" in value:
                raise ValueError("Pass data or legacy state, not both")
            value = dict(value)
            value["data"] = value.pop("state")
            if value["data"] is None:
                raise ValueError("Legacy state cannot be null")
        return value

    @model_validator(mode="after")
    def require_evidence(self) -> Self:
        if self.data is None and self.context is None:
            raise ValueError("Provide data, context, or both")
        return self

    @property
    def state(self):
        """Compatibility accessor; serialized requests use data."""
        return self.data

    @state.setter
    def state(self, value):
        # Validate before assignment so invalid legacy updates leave evidence unchanged.
        updated = type(self).model_validate(
            {
                **self.model_dump(exclude={"data"}),
                "state": value,
            }
        )
        self.data = updated.data
        self.__pydantic_fields_set__.add("data")

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Map legacy updates to data while preserving normal Pydantic copy semantics."""
        if update is not None and "state" in update:
            if "data" in update:
                raise ValueError("Pass data or legacy state, not both")
            if update["state"] is None:
                raise ValueError("Legacy state cannot be null")
            mapped = dict(update)
            mapped["data"] = mapped.pop("state")
            # Pydantic copies normally trust updates; validate the legacy evidence
            # before it becomes the canonical field instead of silently ignoring it.
            validated = type(self).model_validate({**self.model_dump(), **mapped})
            update = {**mapped, "data": validated.data}
        return super().model_copy(update=update, deep=deep)

    @field_validator("data", mode="before")
    @classmethod
    def normalize_messages(cls, value):
        return normalize_state(value)


class AnswerBase(BaseModel):
    # Ignore future response metadata; validate every field used by routing.
    model_config = ConfigDict(hide_input_in_errors=True)
    calibrated: bool | None = None


class NoulAnswer(AnswerBase):
    type: Literal["noul"]
    noul: Probability


class DistributionAnswer(AnswerBase):
    confidence: Probability
    probabilities: Annotated[dict[str, Probability], Field(min_length=2)]

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        if not math.isclose(sum(self.probabilities.values()), 1, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        return self


class ChoiceAnswer(DistributionAnswer):
    type: Literal["choice"]
    choice: str

    @model_validator(mode="after")
    def valid_choice(self) -> Self:
        if self.choice not in self.probabilities:
            raise ValueError("Choice is missing from probabilities")
        if self.probabilities[self.choice] < max(self.probabilities.values()) - 1e-5:
            raise ValueError("Choice must be a most likely option")
        return self


class MultiChoiceAnswer(AnswerBase):
    type: Literal["multi_choice"]
    multi_choice: list[str]
    probabilities: Annotated[dict[str, Probability], Field(min_length=2)]

    @model_validator(mode="after")
    def valid_choices(self) -> Self:
        expected = [key for key, value in self.probabilities.items() if value >= 0.5]
        if self.multi_choice != expected:
            raise ValueError("Multi-choice selections must match the 0.5 probability cutoff")
        return self


class ScoreAnswer(DistributionAnswer):
    type: Literal["score"]
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    legend: dict[str, str]

    @model_validator(mode="after")
    def valid_score(self) -> Self:
        keys = {str(i) for i in range(len(self.legend))}
        if set(self.legend) != keys or set(self.probabilities) != keys:
            raise ValueError("Score levels must be consecutive from zero")
        expected = sum(int(key) * value for key, value in self.probabilities.items())
        if not math.isclose(self.score, expected, abs_tol=1e-5):
            raise ValueError("Score must equal the expected level")
        return self


Answer = Annotated[NoulAnswer | ChoiceAnswer | MultiChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Literal[0]


class DecisionResponse(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    id: str
    model: ModelName
    answers: dict[str, Answer]
    usage: Usage
    request_id: str
    evaluation_time_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, ChoiceAnswer)}

    @property
    def multi_choices(self) -> dict[str, MultiChoiceAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, MultiChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {key: value for key, value in self.answers.items() if isinstance(value, ScoreAnswer)}

    @property
    def usage_metadata(self) -> dict[str, int]:
        return {
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
        }

    def validate_for(self, request: DecisionRequest) -> None:
        if self.model != request.model:
            raise ValueError("Response model does not match the requested model")
        if self.answers.keys() != request.questions.keys():
            raise ValueError("Answer names do not match the submitted questions")
        for name, question in request.questions.items():
            answer = self.answers[name]
            if answer.type != question.type:
                raise ValueError("Answer type does not match the submitted question")
            if isinstance(question, ChoiceQuestion):
                if not isinstance(answer, (ChoiceAnswer, MultiChoiceAnswer)) or set(
                    answer.probabilities
                ) != set(question.criteria):
                    raise ValueError("Answer options do not match the submitted criteria")
            elif isinstance(question, ScoreQuestion):
                expected = {str(i): text for i, text in enumerate(question.criteria)}
                if not isinstance(answer, ScoreAnswer) or answer.legend != expected:
                    raise ValueError("Score legend does not match the submitted criteria")


class Choice(ChoiceQuestion):
    type: Literal["choice"] = "choice"


class MultiChoice(ChoiceQuestion):
    type: Literal["multi_choice"] = "multi_choice"


Noul = NoulQuestion
Score = ScoreQuestion
