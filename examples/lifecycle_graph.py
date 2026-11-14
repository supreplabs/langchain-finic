"""Checkpointed model creation, scoring, and calibration; requires the middleware extra.

This module only builds a graph; it does not create remote models on import.
The caller authorizes writes, supplies synthetic labeled records and record IDs,
keeps the client open, and passes a checkpointer and thread_id. Each write has
its own node; no automatic write retries are configured. Resume a failed wait
with graph.ainvoke(None, config). In-memory checkpoints do not survive process
exit; use a durable checkpointer in a service. An ambiguous write failure still
requires reconciliation: checkpoints cannot make remote writes exactly-once.
"""

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from langchain_finic import (
    CalibrationRequest,
    CreateScoringModelRequest,
    FinicClient,
    FinicScorer,
)


class LifecycleState(TypedDict, total=False):
    name: str
    examples: list[dict]
    record: dict
    outcomes: list[dict]
    model_id: str
    creation_job_id: str
    creation_request_id: str
    creation_ready: bool
    scoring_result: dict
    calibration_job_id: str
    calibration_request_id: str
    calibration_ready: bool


def build_graph(client: FinicClient, checkpointer):
    async def create(state: LifecycleState):
        result = await client.acreate_scoring_model(
            CreateScoringModelRequest(name=state.get("name", "Synthetic"), data=state["examples"])
        )
        return {
            "model_id": result.model_id,
            "creation_job_id": result.job_id,
            "creation_request_id": result.request_id,
        }

    async def wait_creation(state: LifecycleState):
        await client.await_job(state["creation_job_id"], timeout=900)
        return {"creation_ready": True}

    async def score(state: LifecycleState, config: RunnableConfig):
        scorer = FinicScorer(model_id=state["model_id"], client=client)
        result = await scorer.ainvoke(state["record"], config)
        return {"scoring_result": result.model_dump()}

    async def calibrate(state: LifecycleState):
        # Calibration takes labeled records, as creation does, not scored IDs.
        result = await client.acalibrate(
            CalibrationRequest(model_id=state["model_id"], data=state.get("outcomes") or state["examples"])
        )
        return {"calibration_job_id": result.job_id, "calibration_request_id": result.request_id}

    async def wait_calibration(state: LifecycleState):
        await client.await_job(state["calibration_job_id"], timeout=900)
        return {"calibration_ready": True}

    return (
        StateGraph(LifecycleState)
        .add_node("create", create)
        .add_node("wait_creation", wait_creation)
        .add_node("score", score)
        .add_node("calibrate", calibrate)
        .add_node("wait_calibration", wait_calibration)
        .add_edge(START, "create")
        .add_edge("create", "wait_creation")
        .add_edge("wait_creation", "score")
        .add_edge("score", "calibrate")
        .add_edge("calibrate", "wait_calibration")
        .add_edge("wait_calibration", END)
        .compile(checkpointer=checkpointer)
    )
