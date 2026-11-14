import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from langchain_finic import (
    KairosDecisionInput,
    KairosRunnable,
    KairosToolkit,
)


class FakeKairos:
    def __init__(self):
        self.calls = []

    def decide(self, action, properties=None, **kwargs):
        self.calls.append(("decide", action, properties, kwargs))
        return self.decision(action)

    def track(self, event_type, properties=None, **kwargs):
        self.calls.append(("track", event_type, properties, kwargs))

    def track_and_decide(self, event_type, properties=None, **kwargs):
        self.calls.append(("track_and_decide", event_type, properties, kwargs))
        return self.decision(event_type)

    def run_workflow(self, name, input=None, **kwargs):
        self.calls.append(("run_workflow", name, input, kwargs))
        return {"run_id": "run_1", "status": "queued"}

    def flush(self):
        self.calls.append(("flush",))

    @staticmethod
    def decision(action):
        return SimpleNamespace(
            decision="review",
            risk_score=72,
            probability=0.72,
            reasons=["new_device"],
            typology="account_takeover",
            rules=["manual_review"],
            action=f"{action}@1",
            decision_id="decision_1",
            event_id="event_1",
            fallback=False,
            error=None,
        )


def test_kairos_runnable_invokes_typed_decision_and_traces_action():
    client = FakeKairos()
    runnable = KairosRunnable(client=client)

    result = runnable.with_config(tags=["risk-check"]).invoke(
        KairosDecisionInput(action="withdrawal", properties={"user_id": "u1"})
    )

    assert result.decision == "review"
    assert result.risk_score == 72
    assert result.decision_id == "decision_1"
    assert client.calls == [
        (
            "decide",
            "withdrawal",
            {"user_id": "u1"},
            {"flush": True},
        )
    ]


def test_kairos_runnable_works_with_pinned_sdk_without_network():
    kairos_sdk = pytest.importorskip("kairos")
    requests = []

    def transport(method, url, headers, body, timeout):
        requests.append((method, url, headers, json.loads(body)))
        return 200, json.dumps({"decision": "approve", "risk_score": 12, "action": "transfer@1"}).encode(), {}

    client = kairos_sdk.Kairos("synthetic-key", transport=transport)
    try:
        result = KairosRunnable(client=client).invoke(
            {"action": "transfer", "properties": {"user_id": "user_1", "amount": 25}}
        )
    finally:
        client.shutdown()

    assert result.decision == "approve"
    assert result.probability == 0.12
    assert requests[0][0] == "POST"
    assert urlsplit(requests[0][1]).path == "/v1/decide"  # The Kairos SDK's own path, not ours.
    assert requests[0][2]["Authorization"] == "Bearer synthetic-key"
    assert requests[0][3]["action"] == "transfer"
    assert requests[0][3]["entities"] == {"user": "user_1"}
    assert requests[0][3]["properties"] == {"amount": 25}


@pytest.mark.asyncio
async def test_kairos_runnable_ainvoke_uses_same_sdk_contract():
    client = FakeKairos()
    result = await KairosRunnable(client=client).ainvoke(
        {"action": "payout", "properties": {"user_id": "u2"}, "fallback": "review"}
    )

    assert result.decision == "review"
    assert client.calls[0] == (
        "decide",
        "payout",
        {"user_id": "u2"},
        {"fallback": "review", "flush": True},
    )


def test_kairos_runnable_as_tool_returns_json_safe_decision():
    client = FakeKairos()
    tool = KairosRunnable(client=client).as_tool()

    result = tool.invoke({"action": "payment", "properties": {"user_id": "u5"}})

    assert tool.name == "kairos_decide"
    assert result == {
        "decision": "review",
        "risk_score": 72,
        "probability": 0.72,
        "reasons": ["new_device"],
        "typology": "account_takeover",
        "rules": ["manual_review"],
        "action": "payment@1",
        "decision_id": "decision_1",
        "event_id": "event_1",
        "fallback": False,
    }


def test_kairos_toolkit_defaults_to_only_decide():
    client = FakeKairos()

    tools = KairosToolkit(client=client).get_tools()

    assert [tool.name for tool in tools] == ["kairos_decide"]
    assert (
        tools[0].invoke({"action": "payment", "properties": {"user_id": "u3", "amount": 12}})["decision"]
        == "review"
    )
    assert client.calls[0][0:3] == (
        "decide",
        "payment",
        {"user_id": "u3", "amount": 12},
    )


@pytest.mark.asyncio
async def test_kairos_toolkit_async_track_and_decide_and_flush():
    client = FakeKairos()
    tools = KairosToolkit(client=client, operations=("track", "track_and_decide", "flush")).get_tools()
    by_name = {tool.name: tool for tool in tools}

    queued = await by_name["kairos_track"].ainvoke({"event_type": "login", "properties": {"user_id": "u4"}})
    decision = await by_name["kairos_track_and_decide"].ainvoke(
        {
            "event_type": "withdrawal",
            "properties": {"user_id": "u4", "amount": 45},
            "fallback": "review",
        }
    )
    flushed = await by_name["kairos_flush"].ainvoke({})

    assert queued == {"queued": True, "delivery": "background"}
    assert decision["decision"] == "review"
    assert flushed == {"flushed": True}
    assert [call[0] for call in client.calls] == ["track", "track_and_decide", "flush"]


def test_kairos_toolkit_runs_selected_workflow_and_rejects_bad_selection():
    client = FakeKairos()
    tools = KairosToolkit(client=client, operations=("run_workflow",)).get_tools()

    result = tools[0].invoke({"name": "review_payment", "input": {"payment_id": "p1"}})

    assert result == {"run_id": "run_1", "status": "queued"}
    assert client.calls == [("run_workflow", "review_payment", {"payment_id": "p1"}, {"wait": False})]
    with pytest.raises(ValueError, match="at most once"):
        KairosToolkit(client=client, operations=("decide", "decide"))
    with pytest.raises(ValueError):
        KairosToolkit(client=client, operations=("delete",))
