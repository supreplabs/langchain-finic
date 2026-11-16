# Changelog

## 0.1.0b5

- Preserve nested LangChain message normalization in direct, bound, and configurable scorer tools across sync and async calls (#116).
- Add `delivery="unconfirmed"` to Kairos flush tool results. The retained `flushed` field means only that the SDK call returned, not server acknowledgement; monitor SDK `on_error` callbacks for delivery failures (#117).
- Explain outstanding scoring writes after early async batch errors and provide a tested per-record reconciliation example using `return_exceptions=True` (#118).
- Document Kairos async cancellation, remote-write uncertainty, and worker/client shutdown responsibilities (#119).
- Add 28 offline regression cases covering normalization, real-SDK delivery failures, batch reconciliation, and cancellation. Preserve base-only and minimum-dependency coverage without requiring the optional Kairos SDK.

## 0.1.0b4

- Require explicit schema-update and schema-bypass grants for toolkit scoring; preserve LangChain message normalization in sync and async tools.
- Diagnose legacy score-ID-only calibration payloads and clarify onboarding, score evidence, and lifecycle migration.
- Report ambiguous deletion outcomes and actionable lifecycle errors, and expose selected destructive toolkit operations for inspection.
- Add disposable-model evaluation with prevalidated holdouts, recovery ledgers, verified cleanup, baseline metrics, and observed job-wait durations.
- Add reproducible UCI Bank Marketing and Online Shoppers recipes with pinned archives, offline validation, explicit bounded live execution, and attribution.
- Expand offline regression coverage for write timeouts, cleanup verification, cancellation, request budgets, dataset splits, and toolkit permissions.
- Refresh LangChain, langchain-core, LangSmith, LangGraph and Ruff lockfile versions.
- Document a proposal for provider job-progress metadata. Finic still exposes completion/failure only; percentage and ETA support remain pending.

## 0.1.0b3

- Cover the current Grayson ask, scoring-model lifecycle, and calibration API with typed sync/async methods.
- Add explicitly selected LangChain tools for scoring-model operations, keeping permanent deletion opt-in.
- Align score/calibration request models, response contracts, examples, and compatibility monitoring with current Finic docs.
- Calibration now sends full labeled records and requires at least 100 rows, including 20 from each outcome; earlier score-ID-only outcome submissions must be migrated to retain and submit source records.

## 0.1.0b2

- Refresh locked runtime, development, and test dependencies.

## 0.1.0b1

- Promote the package from alpha to beta after the documented live observation
  period and representative application validation.
- Keep the API explicitly prerelease: beta invites application feedback and does
  not claim production stability or predictive accuracy.
- Exclude Kairos-only checks from the Finic live-test job while retaining their
  separate credential-gated workflow.

## 0.1.0a10

- Add an optional Kairos SDK extra, typed LangChain decision Runnable, and
  explicitly selected event/workflow tools with offline contract coverage.
- Publish a permissive `robots.txt` with the documentation sitemap and verify it
  appears at the generated site root.

## 0.1.0a9

- Update `langchain-core` to 1.7.3, `langsmith` to 0.14.16, and the development
  integration to `langgraph` 1.2.18; update Ruff to 0.17.3 and mypy to 2.4.1.
- Update the pinned `setup-uv` GitHub Action to v10.3.3.

## 0.1.0a8

- Preserve configured, bound and configurable decision models when native tool
  arguments omit `model`; retain explicit request-model overrides.
- Document typed decision-tool argument conversion and credential-free application
  testing, and clarify the structured return type of custom scoring calls.

## 0.1.0a7

- Add a versioned synthetic System 1 evaluation corpus and bounded opt-in runner,
  with frozen held-out configuration, failure-inclusive metrics and explicit limitations.
- Type public middleware and Runnable interfaces, including dictionary inputs and exception-bearing batch results; check consumer type assertions on current and minimum dependencies.

- Include registered tool semantics in screening and add bounded context-builder hooks for routing and screening.
- Explain calibration warm-up, question versioning, application deadlines and shared connection ownership; add an offline-tested async routing example.
- Require explicit scoring-tool opt-in for schema mutation/bypass; direct scoring flags remain available.
  Existing agents that need those flags must construct tools with `allow_schema_update=True`
  and/or `allow_schema_bypass=True`.
- Preserve scoring-tool names and JSON output through bindings and configurable fields.
- Apply scorer concurrency defaults to completion-order batches.

- Record sanitized daily live-test evidence and add installed-package application validation.

## 0.1.0a6

- Add typed sync/async clients for custom scoring, model creation, model listing,
  calibration, and job status, covering all six documented Finic REST endpoints.
- Add `FinicScorer`, explicitly selected lifecycle tools through `FinicToolkit`,
  and checkpointed LangGraph composition examples.
- Use single-attempt scoring/creation/calibration writes and explicit bounded job
  polling; retain caller-assigned score IDs for reconciliation.
- Align decisions with documented `data`/`context` inputs, base-rate boundaries,
  and structured provider errors. Existing `state` inputs remain supported;
  serialized requests now use canonical `data`.
- Preserve legacy state assignment/copy updates and configured or bound decision
  tool conversion, including minimum supported LangChain dependencies.
- Accept documented text outcome labels in JSON and CSV without rewriting inputs.
- Expand opt-in live decision coverage to all question types, bounded batches,
  LangGraph tools, middleware, and data/context inputs (15 checks / 18 requests).
- Add isolated synthetic model lifecycle validation, retain pending score IDs
  before writes, and keep validation active under optimized Python.
- Strengthen offline failure, cancellation, ownership, and partial-batch coverage;
  document compatibility, maintenance, and release-promotion policies.
- Retry exact-version PyPI installation verification with fresh index metadata
  and a finite budget, preserving immutable artifacts and registry hash checks.

The package remains alpha. Model creation and calibration are explicit remote
writes; the separate lifecycle test does not run in ordinary CI. Test coverage
validates API and framework behavior, not fraud-detection accuracy.

## 0.1.0a5

- Add opt-in live LangChain standard tool integration tests alongside sync/async
  classifier smoke tests.
- Clarify independent integration ownership and improve README navigation and examples.
- Generate version-specific release notes and detect prerelease versions automatically.
- Update GitHub Actions for uv setup, artifact upload/download, and Pages configuration/deployment.

## 0.1.0a4

Initial public alpha. See [GitHub Releases](https://github.com/supreplabs/langchain-finic/releases)
for publication dates and distribution checksums.

- Typed `Noul`, `Choice`, `MultiChoice`, and `Score` decisions with response validation.
- LangChain Runnable composition, tools, sync/async calls, batching, and tracing metadata.
- Optional model-routing and tool-screening middleware for LangChain agents.
- Configurable model name, transport injection, bounded retries, and client lifecycle management.
- MIT license, typed distributions, usage documentation, and API reference.

Configuration uses `FINIC_*` environment variables. Earlier unpublished prototypes
used `GRAYSON_*` aliases; those aliases are not supported by this release.
