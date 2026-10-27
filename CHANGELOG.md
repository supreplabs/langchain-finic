# Changelog

## Unreleased

- Type public middleware and Runnable interfaces, including dictionary inputs and exception-bearing batch results; check consumer type assertions on current and minimum dependencies.

- Include registered tool semantics in screening and add bounded context-builder hooks for routing and screening.
- Explain calibration warm-up, question versioning, application deadlines and shared connection ownership; add an offline-tested async routing example.
- Require explicit scoring-tool opt-in for schema mutation/bypass; direct scoring flags remain available.
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
