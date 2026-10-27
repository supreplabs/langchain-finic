# Changelog

## Unreleased

- Add a six-endpoint coverage matrix and separately opted-in synthetic lifecycle validation.

- Add FinicScorer, selected lifecycle tools, and checkpointed composition examples.
- Accept documented text outcome labels in JSON and CSV without rewriting inputs.

- Add typed sync/async custom scoring, creation, discovery, calibration, and job APIs.
- Use conservative write retries and explicit bounded job polling.
- Preserve legacy state updates and configured/bound decision-tool conversion.

- Align decisions with Finic data/context inputs, retaining state as a compatibility alias.
- Match documented base-rate boundaries and expose structured provider error identifiers.

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
