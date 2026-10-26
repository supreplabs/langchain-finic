import os

import httpx
import pytest

from langchain_finic import Choice, DecisionRequest, MultiChoice, Noul, Score


@pytest.fixture
def request_body():
    return DecisionRequest(
        state={"events": ["Account details changed before an unfamiliar payment"]},
        questions={
            "fits_profile": Noul(instructions="Is the activity consistent with the account profile?"),
            "action": Choice(
                instructions="Choose a review action", criteria={"close": "Close", "escalate": "Escalate"}
            ),
            "indicators": MultiChoice(
                instructions="Which signals apply?", criteria={"device": "New device", "payee": "New payee"}
            ),
            "risk": Score(instructions="Assess risk", criteria=["Low", "Medium", "High"]),
        },
    )


@pytest.fixture
def response_body():
    return {
        "id": "dec_test",
        "model": "grayson-1",
        "answers": {
            "fits_profile": {"type": "noul", "noul": 0.2},
            "action": {
                "type": "choice",
                "choice": "escalate",
                "confidence": 0.8,
                "probabilities": {"close": 0.1, "escalate": 0.9},
            },
            "indicators": {
                "type": "multi_choice",
                "multi_choice": ["device", "payee"],
                "probabilities": {"device": 0.8, "payee": 0.7},
            },
            "risk": {
                "type": "score",
                "score": 1.6,
                "confidence": 0.55,
                "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
                "legend": {"0": "Low", "1": "Medium", "2": "High"},
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
        "request_id": "req_test",
        "evaluation_time_ms": 12.5,
    }


@pytest.fixture
def handler(response_body):
    return lambda _: httpx.Response(200, json=response_body)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Host credentials, model defaults, and tracing never affect offline tests."""
    for name in list(os.environ):
        if name.startswith(("FINIC_", "GRAYSON_", "LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
