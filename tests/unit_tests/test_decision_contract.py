import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from langchain_finic import Choice, DecisionRequest, FinicClassifier, FinicError, MultiChoice, Noul, Score


@pytest.mark.parametrize(
    "evidence",
    [
        {"data": "evidence"},
        {"context": "background"},
        {"data": {}, "context": "context"},
        {"state": [HumanMessage(content="evidence")]},
    ],
)
def test_documented_wire(evidence, response_body):
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        body = dict(response_body, answers={"risk": {"type": "noul", "noul": 0.2}})
        return httpx.Response(200, json=body)

    request = DecisionRequest(**evidence, questions={"risk": Noul(instructions="Risk?")})
    with FinicClassifier(api_key="test", transport=httpx.MockTransport(handle)) as classifier:
        classifier.invoke(request)
        schema = classifier.as_tool().args_schema.model_json_schema()
    assert "state" not in seen[0]
    assert "state" not in schema["properties"]
    assert "data" in schema["properties"] and "context" in schema["properties"]
    assert seen[0] == request.model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    "evidence",
    [
        {},
        {"data": " "},
        {"context": " "},
        {"context": {}},
        {"data": 42},
        {"state": "a", "data": "b"},
        {"state": None},
        {"data": None},
    ],
)
def test_invalid_evidence(evidence):
    with pytest.raises(ValidationError):
        DecisionRequest(**evidence, questions={"risk": Noul(instructions="Risk?")})


@pytest.mark.parametrize("rate", [0, 1, -0.1, 1.1, float("nan")])
def test_strict_base_rates(rate):
    with pytest.raises(ValidationError):
        Noul(instructions="Risk?", base_rate=rate)
    with pytest.raises(ValidationError):
        MultiChoice(instructions="Risk?", criteria={"a": None, "b": None}, base_rate={"a": rate})


@pytest.mark.parametrize("total", [0.98, 0.99, 1, 1.02])
def test_distribution_tolerance(total):
    choice = Choice(
        instructions="Risk?", criteria={"a": None, "b": None}, base_rate={"a": total / 2, "b": total / 2}
    )
    assert sum(choice.base_rate.values()) == pytest.approx(1)
    score = Score(instructions="Risk?", criteria=["Low", "High"], base_rate=[total / 2, total / 2])
    assert sum(score.base_rate) == pytest.approx(total)


@pytest.mark.parametrize("total", [0.9799, 1.0201])
def test_distribution_outside_tolerance(total):
    with pytest.raises(ValidationError):
        Choice(
            instructions="Risk?", criteria={"a": None, "b": None}, base_rate={"a": total / 2, "b": total / 2}
        )
    with pytest.raises(ValidationError):
        Score(instructions="Risk?", criteria=["Low", "High"], base_rate=[total / 2, total / 2])


def test_structured_error_without_body(request_body):
    def handle(request):
        return httpx.Response(
            400,
            headers={"X-Request-Id": "req_bad"},
            json={
                "error": {
                    "type": "invalid_request",
                    "param": "questions.risk.base_rate",
                    "message": "private customer evidence",
                }
            },
        )

    with FinicClassifier(api_key="secret", transport=httpx.MockTransport(handle)) as model:
        with pytest.raises(FinicError) as exc:
            model.invoke(request_body)
    assert exc.value.error_type == "invalid_request"
    assert exc.value.param == "questions.risk.base_rate"
    assert exc.value.request_id == "req_bad"
    assert exc.value.status == 400
    assert "private" not in str(exc.value) and "secret" not in str(exc.value)
