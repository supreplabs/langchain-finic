"""Explicitly authorized, synthetic model lifecycle; never run by default CI.

Requires FINIC_LIFECYCLE_TESTS=1 and FINIC_API_KEY. One invocation creates one
persistent server-side model. See tests/README.md before running.
"""

import asyncio
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from langchain_finic import CalibrationRequest, CreateScoreRequest, FinicClient, ScoringRequest


def synthetic_examples() -> list[dict]:
    return [
        {
            "amount": 20 + i * 11,
            "account_age_days": 10 + i * 7,
            "new_device": i % 2 == 0,
            "is_bad": i % 2 == 0,
        }
        for i in range(100)
    ]


async def exercise_lifecycle(client: FinicClient, record: Callable[[dict], None]) -> dict:
    """Exercise only the model created here; persist job and score identities before waits and writes."""
    evidence: dict = {"started_at": datetime.now(UTC).isoformat(), "status": "running"}
    created = await client.acreate_score(
        CreateScoreRequest(
            data=synthetic_examples(),
            description=(
                "langchain-finic synthetic integration check. "
                "is_bad is an artificial test label; not a production model."
            ),
        )
    )
    evidence.update(
        model_id=created.model_id, creation_job_id=created.job_id, creation_request_id=created.request_id
    )
    record(evidence)
    job = await client.await_job(created.job_id, timeout=900)
    if not job.succeeded:
        raise RuntimeError("Creation job did not succeed")
    evidence["creation_complete"] = True
    record(evidence)
    listing = await client.alist_scores()
    if not any(item.model_id == created.model_id and item.status == "ready" for item in listing.scores):
        raise RuntimeError("Created model is not listed as ready")
    scores = []
    outcomes = []
    evidence["score_attempts"] = []
    for label in (False, True):
        score_id = "integration_" + uuid4().hex
        attempt = {"model_id": created.model_id, "score_id": score_id, "status": "pending"}
        evidence["score_attempts"].append(attempt)
        # A timeout may follow an accepted write. Keep the same ID for reconciliation.
        record(evidence)
        result = await client.ascore(
            ScoringRequest(
                model_id=created.model_id,
                score_id=score_id,
                data={
                    "amount": 200 if label else 20,
                    "account_age_days": 10 if label else 800,
                    "new_device": label,
                },
            )
        )
        attempt.update(status="received", response=result.model_dump())
        record(evidence)
        if result.score_id != score_id or result.model_id != created.model_id:
            raise RuntimeError("Score response identity does not match the request")
        if not (0 <= result.score <= 1000 and result.request_id):
            raise RuntimeError("Score response is missing a valid score or request ID")
        attempt["status"] = "validated"
        scores.append(result.model_dump())
        outcomes.append({"score_id": score_id, "is_bad": label})
        evidence["scores"] = scores
        record(evidence)
        await asyncio.sleep(0.3)  # Remain below the documented organization rate.
    calibration = await client.acalibrate(CalibrationRequest(model_id=created.model_id, data=outcomes))
    evidence.update(
        calibration_job_id=calibration.job_id,
        calibration_request_id=calibration.request_id,
        matched=calibration.matched,
        unmatched=calibration.unmatched,
    )
    record(evidence)
    if calibration.matched != 2 or calibration.unmatched != 0:
        raise RuntimeError("Calibration did not match both score outcomes")
    calibration_job = await client.await_job(calibration.job_id, timeout=900)
    if not calibration_job.succeeded:
        raise RuntimeError("Calibration job did not succeed")
    evidence.update(status="passed", finished_at=datetime.now(UTC).isoformat())
    record(evidence)
    return evidence


async def run_live(output: Path) -> dict:
    if os.getenv("FINIC_LIFECYCLE_TESTS") != "1" or not os.getenv("FINIC_API_KEY"):
        raise RuntimeError("Set FINIC_LIFECYCLE_TESTS=1 and FINIC_API_KEY after authorizing model creation")
    if output.exists():
        raise FileExistsError("Choose a new evidence path; do not overwrite a previous lifecycle attempt")
    evidence: dict = {"status": "starting", "request_count": 0}

    def record(update: dict) -> None:
        evidence.update(update)
        output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: evidence[key]
                    for key in (
                        "status",
                        "model_id",
                        "creation_job_id",
                        "calibration_job_id",
                        "request_count",
                    )
                    if key in evidence
                }
            ),
            flush=True,
        )

    async def count(request: httpx.Request) -> None:
        evidence["request_count"] += 1
        # Persist before sending: an interrupted/ambiguous write must remain visible.
        record({})

    record({})
    try:
        async with httpx.AsyncClient(event_hooks={"request": [count]}) as http:
            async with FinicClient(max_attempts=1, http_async_client=http) as client:
                return await exercise_lifecycle(client, record)
    except BaseException as exc:
        record(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "error_type": getattr(exc, "error_type", None),
                "http_status": getattr(exc, "status", None),
                "request_id": getattr(exc, "request_id", None),
            }
        )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_live(args.output))
