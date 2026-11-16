"""Explicit disposable-model trial; importing this module performs no remote writes.

Call run_trial only after authorizing creation, scoring, recalibration and permanent
cleanup. The caller supplies disjoint datasets and keeps its FinicClient open.
The ledger contains IDs and aggregate metrics, never input records or credentials.
An ambiguous create/calibration/score write stops automatic cleanup for manual
reconciliation; never restart a trial using a fresh ledger to bypass that state.
"""

import json
from pathlib import Path
from uuid import uuid4

from langchain_finic import (
    CalibrationRequest,
    CreateScoringModelRequest,
    FinicClient,
    FinicError,
    ScoringRequest,
)


def summarize(scores: list[int], labels: list[bool], threshold: int, prior: float) -> dict:
    """Compare fixed decisions and normalized scores with a training-prior baseline."""
    n = len(labels)
    predictions = [score >= threshold for score in scores]
    return {
        "records": n,
        "threshold": threshold,
        "brier": sum((s / 1000 - y) ** 2 for s, y in zip(scores, labels, strict=True)) / n,
        "training_prior_brier": sum((prior - y) ** 2 for y in labels) / n,
        "review_fraction": sum(predictions) / n,
        "true_positive": sum(p and y for p, y in zip(predictions, labels, strict=True)),
        "false_positive": sum(p and not y for p, y in zip(predictions, labels, strict=True)),
        "true_negative": sum(not p and not y for p, y in zip(predictions, labels, strict=True)),
        "false_negative": sum(not p and y for p, y in zip(predictions, labels, strict=True)),
    }


async def run_trial(
    client: FinicClient,
    *,
    training: list[dict],
    calibration: list[dict],
    holdout: list[dict],
    criteria: str,
    threshold: int,
    ledger_path: Path,
) -> dict:
    """Train, evaluate, recalibrate, reevaluate and delete this run's model.

    Requires an admin-created key for deletion. Use max_attempts=1. Holdout rows
    contain full features plus boolean is_bad labels, stripped before scoring.
    Dataset separation and prediction-time feature validity are caller obligations.
    A finished ledger records both trial status and cleanup status independently.
    """
    if not holdout or any(type(row.get("is_bad")) is not bool for row in holdout + training):
        raise ValueError("Supply nonempty holdout and boolean is_bad labels")
    if type(threshold) is not int or not 0 <= threshold <= 1000:
        raise ValueError("Choose an integer threshold from 0 to 1000 before evaluating")
    run_id = uuid4().hex
    creation = CreateScoringModelRequest(name=f"Disposable trial {run_id}", criteria=criteria, data=training)
    # Validate calibration before any remote write, using a placeholder model ID.
    calibration_request = CalibrationRequest(model_id="pending", data=calibration)
    # Validate every holdout record before creating a remote model. Keep normalized
    # copies so caller mutations during awaits cannot change the evaluated inputs.
    holdout_data = [
        ScoringRequest(
            model_id="pending",
            score_id=f"{run_id}-initial_trained-{index}",
            data={key: value for key, value in row.items() if key != "is_bad"},
            include_questions=True,
        ).data
        for index, row in enumerate(holdout)
    ]
    ledger = {"run_id": run_id, "status": "running", "cleanup": "not_needed", "threshold": threshold}
    ledger_path = Path(ledger_path)
    # Exclusive creation prevents accidental replay using an existing ledger.
    with ledger_path.open("x") as stream:
        json.dump(ledger, stream)

    def save():
        temporary = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(ledger_path)

    async def wait(job_id):
        ledger["active_job"] = job_id
        save()
        await client.await_job(job_id, timeout=900, poll_interval=30)
        ledger.pop("active_job")
        save()

    prior = sum(row["is_bad"] for row in training) / len(training)
    labels = [row["is_bad"] for row in holdout]
    model_id = None
    try:
        ledger["pending_write"] = "create"
        ledger["model_name"] = creation.name
        save()
        created = await client.acreate_scoring_model(creation)
        model_id = created.model_id
        ledger.update(model_id=model_id, creation_job_id=created.job_id, cleanup="required")
        ledger.pop("pending_write")
        save()
        await wait(created.job_id)
        for phase in ("initial_trained", "recalibrated"):
            if phase == "recalibrated":
                calibration_request.model_id = model_id
                ledger["pending_write"] = "calibrate"
                save()
                accepted = await client.acalibrate(calibration_request)
                ledger.update(calibration_job_id=accepted.job_id)
                ledger.pop("pending_write")
                save()
                await wait(accepted.job_id)
            scores = []
            question_counts = []
            for index, data in enumerate(holdout_data):
                score_id = f"{run_id}-{phase}-{index}"
                request = ScoringRequest(
                    model_id=model_id,
                    score_id=score_id,
                    data=data,
                    include_questions=True,
                )
                ledger["pending_write"] = {"phase": phase, "score_id": score_id}
                save()
                result = await client.ascore(request)
                scores.append(result.score)
                question_counts.append(len(result.questions or []))
                ledger.pop("pending_write")
                ledger[phase] = {"completed_scores": len(scores), "question_counts": question_counts}
                save()
            ledger[phase]["metrics"] = summarize(scores, labels, threshold, prior)
            save()
        ledger["status"] = "passed"
    except BaseException as exc:
        ledger.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        if "pending_write" in ledger:
            # Acceptance is unknown. Reconcile saved operation/name/IDs before cleanup.
            ledger["cleanup"] = "manual_reconciliation_required"
        elif model_id:
            try:
                if "active_job" in ledger:
                    job = await client.aget_job_status(ledger["active_job"])
                    if not job.complete:
                        raise RuntimeError("Job remains active; defer cleanup")
                ledger["cleanup"] = "deleting"
                save()
                await client.adelete_scoring_model(model_id)
                try:
                    await client.aget_scoring_model(model_id)
                except FinicError as exc:
                    if exc.status != 404:
                        raise
                else:
                    raise RuntimeError("Deleted model is still retrievable")
                models = await client.alist_scoring_models()
                if any(model.model_id == model_id for model in models.scoring_models):
                    raise RuntimeError("Deleted model remains listed")
                ledger["cleanup"] = "verified_deleted"
            except Exception as exc:
                ledger.update(cleanup="manual_reconciliation_required", cleanup_error_type=type(exc).__name__)
        save()
    if ledger["cleanup"] != "verified_deleted":
        raise RuntimeError("Trial requires manual cleanup; inspect the saved ledger")
    return ledger
