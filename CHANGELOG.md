# Changelog

## 0.1.0a4

First standalone release candidate for PyPI, extracted from the Suprep Labs Finic
agent research workspace. Publication status is shown by PyPI and GitHub Releases.

- MIT license, typed distribution metadata, standalone development tools and docs.
- Standard LangChain tool tests and offline framework contract coverage.
- Wheel/sdist verification, compatibility CI, and OIDC release automation.
- Type-checking fixes for response narrowing and validated middleware inputs.
- HTTPX minimum is 0.28.1, matching the standard-test baseline.
- Existing typed decisions, client, LCEL tools, and optional agent middleware retained.

## 0.1.0a3 — internal workspace release

- FINIC_* configuration, default grayson-1 model, routing and tool-screening middleware.
- Owned-client cleanup for configurable Runnables and cancellation.
- GRAYSON_* environment aliases removed; use FINIC_* variables.
