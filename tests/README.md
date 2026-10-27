# API and LangChain validation

The default suite is offline and blocks network sockets. Mock responses exercise
contracts and failure behavior; they do not establish provider availability or
predictive accuracy. Public API methods exist for all six documented REST
endpoints. Live evidence is recorded separately below.

## Coverage matrix

| Endpoint | Request/response contracts | Offline coverage | Live coverage |
| --- | --- | --- | --- |
| `POST /v1/decide` | model, data/context, legacy state mapping, four question types, base rates; typed answers, usage and identity | `test_classifier.py`, `test_decision_contract.py`, `test_model_config.py`, `test_integration.py`, `test_middleware.py` | Ordinary opt-in decision suite; data/context and context-only checks |
| `POST /v1/score` | model_id, stable score_id, JSON/CSV one-record data, schema flags; integer score, model/record/request IDs | `test_scoring_client.py`, `test_scoring_langchain.py` | Separate synthetic lifecycle |
| `POST /v1/scores` | JSON/CSV examples, outcome minima, optional description; accepted model/job/request IDs | `test_scoring_client.py`, `test_lifecycle_runner.py` | Separate synthetic lifecycle |
| `GET /v1/scores` | no arguments; status, criteria, timestamps, example/outcome counts, bad rate | `test_scoring_client.py`, toolkit standard tests | Separate synthetic lifecycle |
| `POST /v1/calibrate` | model_id, JSON/CSV outcomes, matching optional row model_id; matched/unmatched counts and job/request IDs | `test_scoring_client.py`, `test_lifecycle_runner.py` | Separate synthetic lifecycle |
| `GET /v1/jobs/{job_id}` | escaped job path; identity, complete flag, typed failure; explicit wait helpers | `test_scoring_client.py`, checkpoint-resume test | Creation and calibration polling |

Local JSON validation covers finite values, record size, overall request size,
required labels/fields, and score-ID/description lengths. CSV shape and labels
are checked locally, but provider coercion, dotted-column expansion, and the
resulting record size are server checks. Schema compatibility with a remote model
is also server-side. Multipart file upload, MCP, OAuth, account administration,
and retention tools are outside this REST integration's scope.

| Failure or interface | Evidence |
| --- | --- |
| Schema mismatch, missing models/jobs, duplicate score IDs, calibration already running | Sync/async provider-error tests preserve status/type/param |
| Ambiguous write timeout, 429/503 on writes | Single-attempt tests for all three write operations |
| Read retries, invalid JSON, wrong score/model/job identity, invalid score range/type | Transport/response tests in `test_scoring_client.py` and `test_api_coverage.py` |
| Pending, completed, failed jobs, deadline expiry, sync/async cancellation | Job-wait tests; no new poll after cancellation |
| Connection ownership, secret references and default error redaction | Client, classifier and scorer tests; injected clients remain open |
| Classifier tool | Official LangChain unit/integration contracts in `test_standard.py` |
| Scorer plus four lifecycle tools | 55 official tool checks in `test_scoring_langchain.py` |
| LCEL, schemas, callbacks, batches, async concurrency, serialization, tools | Classifier/scorer composition tests |
| Agent tools and checkpointed lifecycle | Scripted chat model and MockTransport tests; failed wait resumes without repeating creation |

## Ordinary live decision checks

Export `FINIC_API_KEY` and run `make integration_test`. Missing credentials skip
checks. The target sets `FINIC_LIVE_TESTS=1` and selects only the `live` marker.
This branch's suite has nine checks / eight paid decision requests, with one
attempt each. A skip is not a successful provider validation. The separate beta
coverage PR #16 adds further checks; reconcile the request count when merging it.

## Separate model lifecycle validation

This is an intentional remote write test, **not** a default CI or daily smoke test.
It creates one persistent model from 100 synthetic examples (50 of each label),
waits for readiness, lists it, scores two records with distinct generated IDs,
submits both outcomes, and waits for calibration. It never accepts a production
model ID. Imports and offline tests perform no remote writes.

After authorizing the run, use a new evidence path:

```bash
export FINIC_API_KEY='your-key'
FINIC_LIFECYCLE_TESTS=1 FINIC_LIFECYCLE_OUTPUT=/tmp/finic-lifecycle-new.json make lifecycle_test
```

Or run the same procedure directly:

```bash
FINIC_LIFECYCLE_TESTS=1 uv run --locked python scripts/live_lifecycle.py --output /tmp/finic-lifecycle-new.json
```

The pytest entry point skips unless both the separate lifecycle flag and key are
set. The direct runner refuses to run without them. The ordinary `FINIC_LIVE_TESTS`
flag alone is insufficient. PR CI and the daily workflow never select this suite.

Budget and effects:

- One creation, one listing, two scoring writes, one calibration: five non-poll
  requests plus at least one poll for each job (seven total minimum).
- Each job waits at most 900 seconds with 30-second polling. At most 30 polls per
  job under that deadline: up to 65 HTTP attempts for a completed run; no automatic
  HTTP retries. Typical creation is documented as 5–10 minutes, but can vary.
- Scoring may consume credits. Finic currently documents creation/calibration as
  free, but they consume quotas (100 creations/organization/day, 10 calibrations/
  model/day). Check [current provider pricing](https://docs.finic.ai/pricing) and
  limits before running. A suite failure does not roll back accepted work.
- Model/job/request/score IDs, status, counts, and scores are saved to the evidence
  file. Credentials, customer records, and provider error messages are omitted.
  Existing files are never overwritten. Store evidence according to your project policy.
- Finic documents no deletion endpoint. A model remains after success or failure.
  A deadline or interrupted write does not authorize creating another model.
  Inspect the saved IDs or the Finic dashboard before deciding what to do next.

Assertions check accepted contracts, identities, score range, matched outcomes,
and job success. They deliberately do not assert exact scores or claim accuracy.

## Recorded provider validation

On 2026-10-27, an explicitly authorized run of `scripts/live_lifecycle.py` passed
against Finic using Python 3.12. It completed in about 97 seconds and made 10 HTTP
requests: the five non-poll requests plus five polls. Both outcome rows matched,
and creation/calibration jobs both succeeded. The evidence file retained full
identifiers; abbreviated references are model `scm_…Q2G`, creation job `job_…4G7`,
and calibration job `job_…PWH`. No further models were created for validation.

Two targeted real decision calls also passed on that date: data plus context,
and context only. The ordinary LangChain/agent failure-path checks remain offline;
these live results should not be described as a live test of every field or an
independent accuracy evaluation.
