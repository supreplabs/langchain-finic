---
description: Finic API compatibility rubric and automated upstream change monitoring.
---

# Finic API compatibility matrix

This matrix reviews the Finic API reference against the package, independently
of LangChain and Python version compatibility. It is a point-in-time assessment,
not a promise that a provider operation is available to every account or that its
predictions meet a customer's requirements. Reviewed on 13 November 2026 against
[Finic's API reference](https://docs.finic.ai/reference).

| Status | Rubric |
| --- | --- |
| Supported | The documented operation has a public package interface and its relevant contract is covered by tests. |
| Partial | A related interface exists, but the current path, request shape, behavior, or test evidence needs review. |
| Missing | The current documented operation has no corresponding public package interface. |
| Unknown | Source retrieval, parsing, or contract review failed; do not count this as compatible. |

| Finic operation | Status | Package coverage and gap |
| --- | --- | --- |
| `POST /v1/ask` | Supported | `FinicClassifier` and `FinicClient.decide` call `/v1/ask` directly. Typed questions, base rates, answers, usage and identity are covered by unit tests. |
| `POST /v1/score` | Supported | `FinicScorer` and `FinicClient.score` send the documented JSON fields, including `include_questions`, and parse the returned questions. Multipart is an alternative encoding the package does not need. |
| `POST /v1/scoring_models` | Supported | `FinicClient.create_scoring_model` creates from criteria alone or from labeled records, and reads both the `201` and `202` response shapes. |
| `GET /v1/scoring_models` | Supported | `FinicClient.list_scoring_models` reads the documented `scoring_models` listing. |
| `GET /v1/scoring_models/{model_id}` | Supported | `FinicClient.get_scoring_model` reads one model with its questions, sources and weights. |
| `PATCH /v1/scoring_models/{model_id}` | Supported | `FinicClient.update_scoring_model` renames, re-states criteria, and archives or unarchives. |
| `DELETE /v1/scoring_models/{model_id}` | Supported | `FinicClient.delete_scoring_model` is exposed and documented as irreversible. `FinicToolkit` never selects it by default, and `FinicToolkit.destructive` names it when an application does. |
| `POST /v1/scoring_models/{model_id}/questions` | Supported | `FinicClient.add_questions` adds up to 20 questions and reads the model with its `added` list. |
| `DELETE /v1/scoring_models/{model_id}/questions/{question_id}` | Supported | `FinicClient.remove_question` removes one question and reads the updated model. |
| `POST /v1/calibrate` | Supported | `FinicClient.calibrate` sends labeled records with `add_questions` and `remove_questions`, and reads `job_id` and `rows`. |
| `GET /v1/jobs/{job_id}` | Supported | Sync and async job status methods are covered by unit tests. |

Supported means the documented request and response contract has a public package
interface covered by offline tests against the documented shapes. It does not mean
every operation has been exercised against a live account: the credential-gated
lifecycle runner covers creation, listing, reading, scoring, calibration and
archiving, and deletion is deliberately left to the application.

Finic records `POST /v1/decide` as the earlier name of `/v1/ask`, and `examples` as
the earlier name of `data` in `calibrate`. The package uses the current names.
Kairos is a separate API and is outside this matrix.

**One open question.** The package offers four question types, and the current
`ask` reference documents three: `noul`, `choice` and `multi_choice`. The fourth,
`score`, with its ordered levels and expected-level answer, was exercised against
`/v1/decide` by the live decision suite as recently as 13 November 2026, so it
works on the earlier path. Whether `/v1/ask` validates question types more strictly
than `/v1/decide` is not something the published reference answers, and offline
tests cannot settle it. The live decision suite exercises a `score` question on
every run, so the first scheduled run after this change is the check that matters.
If it fails, the question type is the thing to look at first.

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
