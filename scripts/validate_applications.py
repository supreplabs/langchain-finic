"""Validate three synthetic consumers of an installed alpha (five live decisions).

Run outside the checkout with FINIC_API_KEY, FINIC_APPLICATION_TESTS=1 and an
installed langchain-finic[middleware]. No model creation or calibration occurs.
Host chat responses are scripted; Finic decisions are live. This checks integration
behavior, not predictive accuracy or independent-user adoption.
"""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import httpx
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

import langchain_finic
from langchain_finic import Choice, DecisionRequest, DecisionResponse, FinicClassifier, Noul
from langchain_finic.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Payment(BaseModel):
    case_id: str
    amount_inr: float = Field(gt=0)
    account_age_days: int = Field(ge=0)
    new_device: bool
    usual_amount_inr: float = Field(gt=0)


class Review(BaseModel):
    case_id: str
    action: Literal["routine", "investigate"]
    request_id: str = Field(min_length=1)


def request(payment):
    return DecisionRequest(
        data=payment.model_dump(),
        context="Entirely fictional payment and customer history, created for software validation.",
        questions={
            "risk": Noul(instructions="Does this payment warrant fraud investigation?"),
            "action": Choice(
                instructions="Choose a review action.",
                criteria={"routine": "Continue routine review", "investigate": "Investigate the payment"},
            ),
        },
    )


async def payment_review(classifier):
    payments = [
        Payment(
            case_id="synthetic_routine",
            amount_inr=200,
            account_age_days=800,
            new_device=False,
            usual_amount_inr=250,
        ),
        Payment(
            case_id="synthetic_unusual",
            amount_inr=50000,
            account_age_days=8,
            new_device=True,
            usual_amount_inr=500,
        ),
    ]
    requests = [request(payment) for payment in payments]
    results = await classifier.abatch(requests, {"max_concurrency": 2})
    require(len(results) == len(payments), "Incomplete payment batch")
    for payment, payload, result in zip(payments, requests, results, strict=True):
        result.validate_for(payload)
        Review(case_id=payment.case_id, action=result.choices["action"].choice, request_id=result.request_id)
    return {"validated_records": len(results), "input_profile": "synthetic customer and transaction context"}


async def checkpoint_review(classifier):
    decision_tool = classifier.as_tool()
    payload = request(
        Payment(
            case_id="synthetic_checkpoint",
            amount_inr=1500,
            account_age_days=500,
            new_device=False,
            usual_amount_inr=1000,
        )
    )
    tool_calls = 0

    async def decide(state: MessagesState):
        nonlocal tool_calls
        tool_calls += 1
        call = state["messages"][-1].tool_calls[0]
        result = await decision_tool.ainvoke({**call, "type": "tool_call"})
        return {"messages": [result]}

    def finish(state: MessagesState):
        message = state["messages"][-1]
        require(isinstance(message, ToolMessage) and message.status == "success", "Tool execution failed")
        require(message.tool_call_id == "checkpoint_case", "Tool identity was lost")
        result = DecisionResponse.model_validate_json(message.content)
        result.validate_for(payload)
        require(bool(result.request_id), "Missing provider request identity")
        return {"messages": [AIMessage(content="Synthetic review recorded")]}

    graph = (
        StateGraph(MessagesState)
        .add_node("decide", decide)
        .add_node("finish", finish)
        .add_edge(START, "decide")
        .add_edge("decide", "finish")
        .add_edge("finish", END)
        .compile(checkpointer=InMemorySaver(), interrupt_before=["finish"])
    )
    config = {"configurable": {"thread_id": "synthetic_checkpoint"}}
    await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": decision_tool.name,
                            "args": payload.model_dump(mode="json"),
                            "id": "checkpoint_case",
                        }
                    ],
                )
            ]
        },
        config,
    )
    snapshot = await graph.aget_state(config)
    require(snapshot.next == ("finish",), "Review checkpoint was not saved")
    result = await graph.ainvoke(None, config)
    require(result["messages"][-1].content == "Synthetic review recorded", "Resume did not complete")
    require(tool_calls == 1, "Resume repeated the decision tool")
    return {"checkpoint_resumed": True, "tool_calls": tool_calls, "persistence": "in-process memory"}


class ScriptedChat(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def routed_review(classifier):
    executed = []

    @tool
    def lookup_demo_case(case_id: str) -> str:
        """Read a fictional case from local memory; no external side effects."""
        executed.append(case_id)
        return "Fictional case available for review"

    def chat(label):
        return ScriptedChat(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "lookup_demo_case", "args": {"case_id": "synthetic_only"}, "id": "lookup"}
                    ],
                ),
                AIMessage(content=label),
            ]
        )

    choices = {
        name: ModelChoice(model=chat(name), criteria=criteria)
        for name, criteria in {"fast": "Simple case lookup", "deep": "Complex investigation analysis"}.items()
    }
    agent = create_agent(
        choices["fast"].model,
        tools=[lookup_demo_case],
        middleware=[
            ModelRouterMiddleware(choices=choices, classifier=classifier),
            AutoModeMiddleware(tools=[lookup_demo_case], classifier=classifier),
        ],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Look up synthetic_only using the local demo lookup tool.")]},
        {"recursion_limit": 6},
    )
    route = result["finic_model_route"]
    require(not route["assessment_failed"] and not route["fallback_used"], "Routing assessment failed")
    require(
        route["selected"] in choices and result["messages"][-1].content == route["selected"],
        "Selected model was not used",
    )
    message = next(item for item in result["messages"] if isinstance(item, ToolMessage))
    require(message.tool_call_id == "lookup", "Tool identity was lost")
    if message.status == "error":
        require(
            not executed and message.artifact["assessment_failed"] is False,
            "Guard failed or executed a blocked tool",
        )
        require(0.5 <= message.artifact["risk_probability"] <= 1, "Invalid blocking assessment")
    else:
        require(executed == ["synthetic_only"], "Allowed tool did not execute exactly once")
    return {"route_validated": True, "guard_validated": True, "host_chat_model": "scripted"}


async def run(output: Path, expected_version: str):
    require(
        os.getenv("FINIC_APPLICATION_TESTS") == "1" and bool(os.getenv("FINIC_API_KEY")),
        "Explicit application opt-in and FINIC_API_KEY required",
    )
    require(not output.exists(), "Choose a new evidence path")
    require(version("langchain-finic") == expected_version, "Installed package version differs")
    package_path = Path(langchain_finic.__file__).resolve()
    require("site-packages" in package_path.parts, "Use an installed distribution outside the checkout")
    evidence = {
        "started_at": datetime.now(UTC).isoformat(),
        "package_version": expected_version,
        "source": "maintainer application validation",
        "host_chat_model": "scripted",
        "status": "running",
        "requests": 0,
        "applications": [],
        "versions": {
            name: version(name)
            for name in ("langchain-finic", "langchain-core", "langchain", "langgraph", "pydantic", "httpx")
        },
    }

    def save():
        output.write_text(json.dumps(evidence, indent=2) + "\n")

    async def count(http_request):
        evidence["requests"] += 1
        save()
        require(evidence["requests"] <= 5, "Application request budget exceeded")

    save()
    try:
        async with httpx.AsyncClient(event_hooks={"request": [count]}) as http:
            async with FinicClassifier(
                model="grayson-1", max_attempts=1, http_async_client=http
            ) as classifier:
                for application, budget in [(payment_review, 2), (checkpoint_review, 1), (routed_review, 2)]:
                    item = {"name": application.__name__, "status": "running"}
                    evidence["applications"].append(item)
                    save()
                    before = evidence["requests"]
                    detail = await application(classifier)
                    require(evidence["requests"] - before == budget, "Unexpected application request count")
                    item.update(status="passed", requests=budget, checks=detail)
                    save()
                    await asyncio.sleep(0.4)
        evidence["status"] = "passed"
    except BaseException as exc:
        evidence.update(status="failed", exception_type=type(exc).__name__)
        if evidence["applications"] and evidence["applications"][-1]["status"] == "running":
            evidence["applications"][-1]["status"] = "failed"
        raise
    finally:
        evidence["finished_at"] = datetime.now(UTC).isoformat()
        save()
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="0.1.0a6")
    args = parser.parse_args()
    asyncio.run(run(args.output, args.version))
