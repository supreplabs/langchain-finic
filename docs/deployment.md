# Calibration and latency budgets

Treat a decision's probability, calibration state, and operating threshold as
separate inputs to an application policy. Successful integration tests establish
API behavior; they do not establish detection quality or a latency guarantee.

## Calibration during evaluation

Finic applies a supplied `base_rate` after a question has accumulated 30 answers.
Before then, the response has `calibrated: false`; after application it is `true`.
Questions without a base rate omit the field, represented as `None` by the package.
Check the returned value rather than assuming a particular request completes
warm-up. The running summary receives answers even without a base rate.

The provider identifies a question by its name, type, and option keys (the number
of levels for a score). Changing instructions or descriptions retains its summary;
changing its identity starts a new warm-up. For materially different experimental
questions, use explicit versioned names, such as `payment_risk_v2`. A new name is
not a reset of a custom scoring model. See [Finic's calibration contract](https://docs.finic.ai/concepts/calibration).

Use development records to choose prompts, base rates, thresholds, and escalation
policy. Reserve held-out records for the documented final comparison. Record the
exact questions, model, base rate, request order, calibration flags, warm-up history,
and evaluation time. Reusing a question identity across hill-climbing candidates
can reuse prior calibration; changing identity can compare a warmed candidate
against a cold one. State these conditions instead of attributing every score
change to the prompt. Do not present `calibrated: true` as proof of calibration on
your population: measure reliability and decision errors on representative data.

The [System 1 evaluation work](https://github.com/supreplabs/langchain-finic/issues/36)
tracks quality, uncertainty, cost, and latency comparisons separately from contract
tests. Application-specific acceptance targets need evidence from those evaluations.

## Transport timeouts and application deadlines

The default `FINIC_TIMEOUT_SECONDS` is 60; supported values exceed 30 seconds.
`FINIC_MAX_ATTEMPTS` defaults to three for decisions and reads, with backoff between
retryable attempts. This follows [Finic's retry guidance](https://docs.finic.ai/reference).
An HTTP timeout limits individual transport phases, not the entire user request.
Retries, queueing, classification, host-model calls, and tools all contribute to
end-to-end latency. Scoring, creation, and calibration writes still make one attempt.

An application can impose a shorter decision budget with `asyncio.timeout` and an
explicit fallback. [The executable example](https://github.com/supreplabs/langchain-finic/blob/main/examples/deadline_routing.py)
uses the capable host-model route when the assessment times out, fails, or has low
probability. Its one-second budget and 0.7 threshold are illustrative policy values,
not measured performance claims. It returns a route; it does not invoke a chat model.

```python
import asyncio

from langchain_finic import FinicClassifier

async def classify_with_budget(classifier, request):
    try:
        async with asyncio.timeout(1.0):
            return await classifier.ainvoke(request)
    except TimeoutError:
        return None  # Application explicitly escalates or uses its configured fallback.
```

The deadline covers the awaited decision, including retries. It does not cover a
subsequent host-model run or guarantee an exact wall-clock bound if the event loop
is blocked or cleanup is slow. Caller cancellation propagates; do not swallow
`asyncio.CancelledError` or shield assessment calls. See [Python's timeout and
cancellation behavior](https://docs.python.org/3/library/asyncio-task.html#timeouts).
Canceling local I/O does not undo an accepted remote request. Never use a timeout
as a reason to replay an ambiguous scoring, creation, or calibration write.

## Connection ownership for repeated agent calls

Enter one classifier context for the service or agent lifetime and share it with
middleware. This reuses owned connections and closes them when the context exits:

```python
from langchain_finic import FinicClassifier
from langchain_finic.middleware import AutoModeMiddleware

async def use_guard(protected_tool, run_agent):
    async with FinicClassifier(max_attempts=1) as classifier:
        guard = AutoModeMiddleware(tools=[protected_tool], classifier=classifier)
        return await run_agent(guard)  # Keep every agent invocation inside this scope.
```

An injected classifier remains caller-owned. A middleware-created classifier is
closed by the middleware context, but that context currently does not enter the
classifier's pooling context: its owned connections are scoped to each invocation.
Use the explicit shared context above when repeated calls should reuse a pool.
Keep each async client on one event loop and finish in-flight work before closing.
Pooling reduces connection setup work; measure its actual effect with your payloads,
concurrency, provider region and fallback rate rather than assuming a speedup.
