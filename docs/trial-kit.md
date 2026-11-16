# langchain-finic 0.1.0b5 trial kit

Purpose: reproduce developer onboarding and migration tasks, record actual results,
and distinguish observed execution from static review. Simulated agent sessions
must label their reports synthetic; no attribution to named people.

Record the exact source commit and working-tree changes under review.
When testing an unpublished fix, distinguish it from the installed PyPI release.

Task A: Follow README and docs/usage.md to configure a typed decision and construct
an agent. Record prerequisites, confusing steps, and which commands were executed.
Task B: Follow docs/scoring.md and examples/lifecycle_graph.py to discover models,
request question evidence, migrate calibration records, and construct a checkpointed
workflow. Check score tool schema privileges and legacy payload diagnostics.

Choose and record the execution mode before starting:

- Static review: inspect docs/source without credentials or API calls.
- Authorized live trial: install a wheel outside the checkout in a fresh environment;
  load configured credentials privately and disable external tracing. Use fictional
  inputs, `max_attempts=1`, a request budget, and sanitized evidence. Never print
  environment files, headers or raw sensitive responses. Record whether the wheel
  is published or contains unreleased fixes.

For either mode, report exact references, observed results, and remaining limits.
Do not claim execution for steps only inspected.

For a later human-run trial, create a fresh venv with Python >=3.11, install the
exact PyPI version, configure FINIC_API_KEY outside recorded logs, and run fictional
cases only. Lifecycle writes need explicit approval and leave remote state.

Feedback template:
- Reviewer type: human / simulated agent
- Version, commit or working-tree state, Python and dependency versions
- Task and expected result
- Steps actually executed and result (pass / fail / not executed)
- Friction, file references, reproducible errors, suggested fix
- Limits: credentials, network, real host model, and API calls used or absent
- Human reviewer confirmation, if any; never invent this field

A release candidate can use these findings as maintainer quality evidence.
Independent-user acceptance requires separately recorded actual user feedback.

## Additional live scenarios

1. Submit a context-only decision and validate response identity and question keys.
2. Compose input mapping, classifier and output mapping through LCEL.
3. Invoke the generic tool and parse its result as `DecisionResponse`.
4. Submit an empty questions mapping and verify rejection before HTTP.
5. Batch three distinct questions with concurrency two; check order, response
   identities and unique request IDs.
6. List existing scoring models and record counts without customer/model content.
7. Retrieve an existing model and compare its identity with the listing. If there
   are no models, record a skip rather than creating one implicitly.

Record requests, elapsed time, status and exception type for each scenario.
Model creation, calibration, schema updates and deletion require a separate
explicitly authorized write plan. A successful small sample does not establish
accuracy, throughput, latency guarantees or production resilience.

For the full remote workflow and cleanup procedure, use the
[disposable scoring tutorial](tutorials/disposable-scoring.md). Record initial-trained
and recalibrated results, chosen threshold, baseline, evidence changes, and cleanup
status. For HTTP agents, record actual queue actions plus duplicate, conflicting-ID
and invalid-input behavior; specify persistence and worker limitations.

For pinned UCI sources, deterministic splits and bounded execution, follow the
[public dataset scoring recipes](tutorials/public-dataset-scoring.md).
