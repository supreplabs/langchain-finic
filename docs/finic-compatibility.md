---
description: Finic API compatibility rubric and automated upstream change monitoring.
---

# Finic API compatibility matrix

This matrix reviews the Finic API reference against the package, independently
of LangChain and Python version compatibility. It is a point-in-time assessment,
not a promise that a provider operation is available to every account or that its
predictions meet a customer's requirements. Reviewed on 16 November 2026 against
[Finic's API reference](https://docs.finic.ai/reference).

| Status | Rubric |
| --- | --- |
| Supported | The documented operation has a public package interface and its relevant contract is covered by tests. |
| Partial | A related interface exists, but the current path, request shape, behavior, or test evidence needs review. |
| Missing | The current documented operation has no corresponding public package interface. |
| Unknown | Source retrieval, parsing, or contract review failed; do not count this as compatible. |

| Finic operation | Status | Package coverage and gap |
| --- | --- | --- |
| `POST /v1/ask` | Supported | `FinicClassifier` uses this endpoint; `/v1/decide` remains available as a client compatibility alias. |
| `POST /v1/score` | Supported | `FinicScorer` and `FinicClient.score` validate current score IDs, schema flags, optional question evidence, and response identity. |
| `POST /v1/scoring_models` | Supported | Typed creation supports criteria-only immediate creation and labeled-data asynchronous jobs. |
| `GET /v1/scoring_models` | Supported | `list_scoring_models` returns typed current model summaries. |
| `GET /v1/scoring_models/{model_id}` | Supported | `get_scoring_model` returns model metadata and question weights, checking response identity. |
| `PATCH /v1/scoring_models/{model_id}` | Supported | `update_scoring_model` supports renaming, criteria changes, and reversible archive/unarchive. |
| `DELETE /v1/scoring_models/{model_id}` | Supported | Explicit client method; permanent deletion is not exposed by default in the LangChain toolkit. |
| `POST /v1/scoring_models/{model_id}/questions` | Supported | `add_questions` validates question limits and returns added-question use status. |
| `DELETE /v1/scoring_models/{model_id}/questions/{question_id}` | Supported | `remove_question` returns the updated model and is not automatically replayed. |
| `POST /v1/calibrate` | Supported | Calibration validates labeled-record requirements and supports adding/removing questions. |
| `GET /v1/jobs/{job_id}` | Supported | Sync and async job status methods are covered by unit tests. |

Finic documents `/v1/decide` as an earlier name of `/v1/ask` that still works.
The legacy `/v1/scores` methods remain available for package compatibility but are
not counted as current API coverage; new code should use the `/v1/scoring_models`
methods. Requests use the documented JSON bodies. Finic also accepts multipart
requests for selected endpoints; multipart encoding is not currently exposed.
Kairos is a separate API and is outside this matrix.

## Weekly check

The [Finic compatibility workflow](https://github.com/supreplabs/langchain-finic/actions/workflows/finic-compatibility.yml)
runs each Monday at 03:27 UTC and can be dispatched manually. It reads the provider's
public Markdown reference and each endpoint page, compares operations and page hashes
with the reviewed [baseline](https://github.com/supreplabs/langchain-finic/blob/main/data/finic-compatibility/baseline.json),
and writes a report to the workflow summary. It opens tracking issues for missing or
partial operations, new or removed endpoints, and changed documentation for otherwise
supported endpoints. An existing open issue prevents another for the same finding;
closing one prevents re-creation for the same evidence. Different evidence can open a
new issue. Closed issues are not automatically reopened.

A changed page is a **review signal**, not proof of a breaking change. The check does
not independently verify all schema, behavior, live availability, Kairos, or LangChain
integration compatibility. A failed fetch or unexpected reference format fails the
workflow and reports coverage as unknown instead of claiming that all is well.

When resolving a finding, inspect the current provider documentation and tests, update
this matrix and the matching baseline entry's status, note, link and SHA-256 of the
normalized Markdown page (LF line endings, surrounding whitespace removed), then
close the tracking issue. Review new or removed endpoints in the baseline as well.
Do not update hashes simply to suppress a finding. The monitor uses no Finic key and
makes no calls to the Finic scoring API.

## Kairos SDK compatibility matrix

The adapter borrows a Python SDK client. This matrix covers the operations exposed
by the adapter. See the [Kairos Python SDK guide](https://kairos.finic.ai/docs/reference/clients/python).

| Adapter operation | SDK method | Offline contract checked |
| --- | --- | --- |
| Decide | `decide` | Action, entity and property encoding, typed decision and normalized probability |
| Track event | `track` | Queued result, event batch encoding when flushed |
| Track and decide | `track_and_decide` | Event and entity encoding, structured decision response |
| Run workflow | `run_workflow` | Workflow path, inputs, supplied idempotency key, returned run ID |
| Flush | `flush` | Delivery remains unconfirmed on success and SDK-reported permanent or transient failure |

Pull request CI checks the pinned SDK. The weekly workflow also resolves the
latest SDK allowed by the package's Kairos extra, logs its version, and runs
the adapter tests with the real SDK and a fake transport. A failure appears
in the workflow's Kairos job. The Grayson documentation drift check still
handles issue creation for its own findings. The Kairos job exercises SDK
behavior and does not fingerprint Kairos documentation or require credentials.
