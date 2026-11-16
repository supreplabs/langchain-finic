---
description: Evaluate a disposable Finic scoring model, recalibrate it, and verify cleanup with a saved operation ledger.
---

# Evaluate and clean up a custom scoring model

Creating a model from labeled records already calibrates it to those records.
A later calibration fits it to another labeled dataset. Compare **initial trained**
and **recalibrated** results, rather than calling the first model uncalibrated.
Criteria-only creation is different: it needs questions or calibration before scoring.

The [executable example](https://github.com/supreplabs/langchain-finic/blob/main/examples/disposable_scoring.py)
implements creation, job waiting, two scoring passes, recalibration and verified
deletion. Download that file alongside your application; it is not an import from
the installed package. Importing it makes no calls.

## Prepare and authorize the trial

Prepare disjoint `training`, `calibration` and `holdout` lists. Each row contains
original feature fields and a boolean `is_bad`. For nonpurchase scoring, for example,
`is_bad=True` means no purchase, so a high score means nonpurchase risk. It says
nothing about whether a customer is a bad person or should receive a promotion.
Creation and calibration need at least 100 rows and 20 from each outcome; Finic
uses up to 2,000 rows. The example uses JSON with booleans for unambiguous metrics.

Remove target fields and features unavailable when a decision is needed. Bank call
duration is unavailable before a call. Final session totals cannot establish an
early-session prediction. Keep natural holdout prevalence, document time boundaries,
and audit repeated entities across splits. A row split does not prove customer separation.

The caller explicitly authorizes model creation, scoring, recalibration and permanent
model deletion. Use a key created by an organization admin for cleanup. A 404 from
deleting a nonexistent ID does not establish deletion permission. Budget two scoring
calls per holdout record plus creation, calibration, polling, deletion and verification.
Do not use an existing production model with this example.

```python
import asyncio
from pathlib import Path
from langchain_finic import FinicClient
from disposable_scoring import run_trial

async def evaluate(training, calibration, holdout, threshold):
    async with FinicClient(max_attempts=1) as client:
        return await run_trial(
            client,
            training=training,
            calibration=calibration,
            holdout=holdout,
            criteria="Historical sessions. Bad means no purchase.",
            threshold=threshold,
            ledger_path=Path("trial-ledger.json"),
        )

# Call asyncio.run(evaluate(...)) with your prepared, authorized datasets.
```

## Evaluate a decision policy

Choose the threshold on a separate development set or from a stated business policy
before touching the final holdout. **500 is an illustrative threshold, not a validated
policy.** A model can achieve high accuracy by routing every record into the majority
class. Report the confusion matrix, review fraction, and a constant baseline alongside
accuracy. For a larger holdout, add PR-AUC, ROC-AUC and uncertainty intervals.

The example computes Brier error using `score / 1000`, Finic's approximate bad-outcome
probability for calibrated models, and compares it with the training outcome prior.
A lower Brier error in a tiny sample does not prove useful ranking or business impact.
Keep the same holdout rows and threshold across phases; use distinct score IDs for
each phase. Do not tune a threshold after inspecting the final holdout.

With `include_questions=True`, evidence describes the questions used in that call.
Recalibration can change question membership, count and weights. Retain per-response
question IDs and metadata when needed for audits; do not join evidence by array
position or assume that equal model IDs imply identical question definitions over time.
The example records counts only, to avoid storing potentially sensitive question text.

## Recover and verify cleanup

Each ledger update atomically replaces a local JSON file. The example saves a pending
operation before a write and returned model/job IDs before waiting. Job responses expose
completion and failure rather than percent progress; waits may last several minutes.
A 900-second wait timeout bounds this demonstration, not the provider's training time.

On successful execution the example deletes only its newly created model, then verifies
GET returns 404 and the model is absent from listing. It preserves an original execution
failure even if cleanup also fails, recording cleanup errors separately. A terminal failed
job can be cleaned up; an active job defers cleanup. Check both trial and cleanup status.

If a write times out or response validation fails, remote acceptance may be unknown.
The ledger marks `manual_reconciliation_required`: inspect its unique model name,
model/job/score IDs and provider state. An accepted creation whose response was lost
may need identification by the unique saved name. Do not create another model or
replay a score automatically. Reconcile the job before deletion; verify deletion even
if its response was lost. The file is useful recovery evidence, not an automatic
resume engine or a guarantee against process/power loss. Keep it on durable storage
and use your application's transactional state store for service operation.

## HTTP agents and duplicate requests

The trial services used a serial localhost endpoint, a Finic score node, a fixed policy,
and a local queue action. Their duplicate cache prevented duplicate calls only within
one running process. They were stopped after testing. This is not evidence of restart
recovery, multi-worker idempotency, or exactly-once remote effects.

For a persistent service, store an idempotency key together with a canonical payload
hash, operation state and result in a transactional database. Reject the same key with
a different payload. Use uniqueness constraints to coordinate workers and atomically
commit the result plus an outbox entry for the queue action. Consumers should deduplicate
outbox messages. Keep a stable Finic score ID and reconcile an ambiguous remote response:
a local transaction cannot make the remote HTTP call atomic. Authenticate incoming
requests and return application-safe errors rather than raw provider responses.
