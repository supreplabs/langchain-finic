"""Exercise the public review workflow without either provider's credentials."""

import json
import runpy
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from langchain_finic import FinicClassifier, FinicError

build_review_graph = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "examples/finic_host_review.py")
)["build_review_graph"]


@pytest.mark.parametrize("probability,queue", [(0.2, "standard"), (0.7, "urgent")])
def test_review_workflow_routes_and_drafts(probability, queue):
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={
            "id": "dec_example", "model": payload["model"],
            "answers": {"takeover": {"type": "noul", "noul": probability}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "request_id": "req_example", "evaluation_time_ms": 1,
        })

    chat = FakeMessagesListChatModel(responses=[AIMessage(content="Review the fictional case.")])
    with FinicClassifier(api_key="test", transport=httpx.MockTransport(respond)) as classifier:
        result = build_review_graph(classifier, chat).invoke({"evidence": {"events": ["New device"]}})
    assert len(calls) == 1
    assert calls[0]["data"] == {"events": ["New device"]}
    assert (result["probability"], result["queue"], result["request_id"]) == (
        probability, queue, "req_example"
    )
    assert result["analyst_note"] == "Review the fictional case."


def test_review_workflow_stops_on_decision_failure():
    def fail(_):
        return httpx.Response(401)

    chat = FakeMessagesListChatModel(responses=[AIMessage(content="should not run")])
    with FinicClassifier(api_key="test", transport=httpx.MockTransport(fail), max_attempts=1) as classifier:
        with pytest.raises(FinicError):
            build_review_graph(classifier, chat).invoke({"evidence": {"events": ["New device"]}})
    assert chat.i == 0
