---
description: Finic API compatibility rubric and automated upstream change monitoring.
---

# Finic API compatibility matrix

This matrix reviews the Finic API reference against the package, independently
of LangChain and Python version compatibility. It is a point-in-time assessment,
not a promise that a provider operation is available to every account or that its
predictions meet a customer's requirements. Reviewed on 11 November 2026 against
[Finic's API reference](https://docs.finic.ai/reference).

| Status | Rubric |
| --- | --- |
| Supported | The documented operation has a public package interface and its relevant contract is covered by tests. |
| Partial | A related interface exists, but the current path, request shape, behavior, or test evidence needs review. |
| Missing | The current documented operation has no corresponding public package interface. |
| Unknown | Source retrieval, parsing, or contract review failed; do not count this as compatible. |

| Finic operation | Status | Package coverage and gap |
| --- | --- | --- |
| `POST /v1/ask` | Partial | `FinicClassifier` and `FinicClient.decide` call the documented earlier `/v1/decide` alias. Review direct `/v1/ask` support and its current contract. |
| `POST /v1/score` | Partial | `FinicScorer` and `FinicClient.score` provide JSON scoring. Review current fields and multipart support. |
| `POST /v1/scoring_models` | Missing | Creation uses the older `/v1/scores` path; the current endpoint and criteria-only creation need review. |
| `GET /v1/scoring_models` | Missing | Listing uses the older `/v1/scores` path. |
| `GET /v1/scoring_models/{model_id}` | Missing | Model details and questions have no client method. |
| `PATCH /v1/scoring_models/{model_id}` | Missing | Updating or archiving a model has no client method. |
| `DELETE /v1/scoring_models/{model_id}` | Missing | Deletion has no client method; any implementation needs an explicit destructive-operation policy. |
| `POST /v1/scoring_models/{model_id}/questions` | Missing | Adding questions has no client method. |
| `DELETE /v1/scoring_models/{model_id}/questions/{question_id}` | Missing | Removing questions has no client method. |
| `POST /v1/calibrate` | Partial | A client method exists; review the current labeled-record and question-update contract. |
| `GET /v1/jobs/{job_id}` | Supported | Sync and async job status methods are covered by unit tests. |

Finic documents `/v1/decide` as an earlier name of `/v1/ask` that still works.
The legacy paths used by the package require separate verification; their absence
from the current endpoint table does not establish that they have stopped working.
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
