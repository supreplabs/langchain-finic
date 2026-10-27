# Evaluating routing and tool screening

`scripts/system1_eval.py` evaluates the package's actual middleware hooks against
36 short fictional, reference-labeled scenarios in
`tests/evaluation/system1-v1/corpus.json`. It is separate from API contract tests.
No proposed tool action or host chat model is executed. A scripted host instance
only satisfies router construction; it does not generate answers.

## Corpus and labels

The corpus contains 12 training examples, 12 development examples and 12 held-out
examples. Each split has three cases per label: `fast`, `powerful`, `allow`, `block`.
Cases include follow-ups, opaque tool names with different side effects, read-only
and reversible operations, destructive actions, role-spoofing inside arguments,
and missing authorization. All scenarios are hand-authored fiction; there are no
customer records. Provenance and the exact label policy are stored with the data.

Routing labels express a **task policy**: simple lookup, extraction, arithmetic
and formatting use `fast`; ambiguity, conflicting evidence and multi-constraint
reasoning use `powerful`. They are not measurements of any named host model's
capabilities. Screening labels describe whether the exact proposed action is
supported by the supplied user request and trusted tool semantics.

Training examples explain that policy; the runner never calls Finic on them and
performs no model training. Every case's label and rationale stay outside the
request sent to Finic. The router receives the short conversation selected by its
context builder. The guard receives its selected conversation, registered tool
description/schema and proposed call. It never calls the real tool.

## Running a bounded evaluation

Run from a checkout with the middleware extra and the current context-hook APIs:

```bash
uv sync --locked --group test --extra middleware
export FINIC_API_KEY=your_key
export FINIC_EVALUATION_TESTS=1
uv run --locked --group test python scripts/system1_eval.py \
  --split development --max-requests 12 \
  --output artifacts/evaluation/development-v1.json
```

This makes **12 decision requests**, at most four request starts per second,
sequentially, with `max_attempts=1`. It creates no score models, submits no outcomes,
and performs no tool side effects. Other services sharing the same organization
still share Finic's rate limit; the runner is not an organization-wide limiter.
Request counts are observed in an HTTPX request hook, including failed attempts.
The hook enforces 250 ms spacing immediately before transport, after request
construction and checkpoint writes.

Use development results for question/context or threshold experimentation. Each
run needs a new output filename. After freezing the implementation, dependencies,
corpus and configuration, evaluate the held-out split:

```bash
uv run --locked --group test python scripts/system1_eval.py \
  --split heldout --max-requests 12 \
  --freeze-from artifacts/evaluation/development-v1.json \
  --output artifacts/evaluation/heldout-v1.json
```

This makes **another 12 requests**. Held-out execution requires a completed,
error-free development report matching the implementation hash, dependency
versions, corpus hash and exact configuration. Threshold/model/seed/price changes
must first produce a new development report. The CLI does not tune thresholds.

This mechanism detects configuration drift; it cannot stop a person reading or
repeatedly inspecting held-out labels. Do not optimize against the held-out
results. After inspecting them, treat that version as an audit set: collect new
cases before another tuning cycle. For repeatability experiments, preregister the
number and order of repeats rather than choosing favorable runs.

## Metrics and interpretation

Reports retain case IDs, request IDs, predictions/probabilities, calibration flags,
latencies, observed HTTP attempts, versions, configuration, deterministic shuffled
case order and corpus/source hashes. They exclude request payloads, credentials
and raw exception messages. Partial progress is written after each completed case;
HTTP attempts are persisted before transport. Interruptions preserve the report.
An interrupted case retains its attempt count without inventing a prediction;
it counts as a failed assessment and is excluded from routing/screening decision
metrics. Existing output files are never overwritten by a new run.

- **Routing:** accuracy against policy labels, under/overprovisioning, and
  separate fallback and powerful-selection rates; compare with always selecting
  `powerful`. Powerful selection includes both explicit decisions and fallbacks.
- **Screening:** true/false blocks, unsafe allows, precision, recall and false-block
  and unsafe-allow rates. Provider errors count as failed assessments and use the
  middleware's fail-closed/fallback behavior; they do not disappear from denominators.
- **Probabilities:** Brier score and five calibration bins on successful screening
  responses only, with the sample count visible. Failed requests have no invented
  probability. These tiny bins are descriptive, not calibration certification.
- **Uncertainty:** proportions include two-sided 95% Wilson intervals. Three unsafe
  cases per split are nowhere near enough to establish a low production failure rate.
- **Latency:** p50/p95 of each complete middleware decision, including local request
  construction, HTTP/retries policy and request-hook pacing sleep. This is not
  full host-agent latency. There is no host-answer quality or host-cost measurement;
  those report fields remain `null`.
- **Cost:** reported successful input tokens multiplied by the configurable
  `--input-usd-per-million` price (default USD 0.035). This is an estimate, not a bill.
  `unknown_usage_attempts` counts attempts without successful usage records.
  `cost_complete` requires a complete run, no failed assessments and a successful
  usage record covering every observed attempt. Interrupted runs are incomplete
  even if interrupted before their first response.

The runner sends no `base_rate`; `calibrated` is recorded, typically absent/null.
It does not claim deployment calibration. If adding base rates later, revise the
protocol to account for [Finic's warm-up and question identity rules](https://docs.finic.ai/concepts/calibration).
Confirm [current pricing](https://docs.finic.ai/pricing) before a paid run.

## Baseline and release interpretation

`tests/evaluation/system1-v1/reference-baseline.json` records zero-call reference
baselines derived from labels: always choosing `powerful` has routing accuracy
3/6 (50%) per split; always blocking has screening recall 3/3 and false-block
rate 3/3. Those are deliberately simple comparison policies, not Finic results.
They do not measure host-model task success, cost or latency.

Initial investigation targets, fixed before the first provider run:

- No failed assessments.
- Routing accuracy above the 50% fixed-powerful policy baseline, with all
  underprovisioned cases inspected.
- Zero unsafe allows on the three unsafe scenarios; inspect every false block.
- Record latency and cost, without an invented production SLO or savings claim.

These targets are diagnostic; satisfying them does **not** establish production
readiness. Before such a claim, expand independently labeled held-out data, validate
actual host-model task success against a fixed-model agent, measure total end-to-end
latency/cost, evaluate domain-specific error costs, and run realistic adversarial and
load tests. The report always marks production readiness as unproven. Existing beta
observation and independent-user/upgrade requirements remain separate.
