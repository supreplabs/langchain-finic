# Compatibility and support

The published package is currently alpha. This policy identifies the supported
surface and the commitments planned for beta and the first final `0.1.x` series.
It does not change the current release's maturity status.

## Public imports

Prefer these imports when building applications:

| Module | Supported names |
| --- | --- |
| `langchain_finic` | `FinicClassifier`, `FinicClient`, `FinicError` |
| `langchain_finic` | `DecisionRequest`, `DecisionResponse` |
| `langchain_finic` | `Noul`, `Choice`, `MultiChoice`, `Score` |
| `langchain_finic` | `NoulAnswer`, `ChoiceAnswer`, `MultiChoiceAnswer`, `ScoreAnswer` |
| `langchain_finic.middleware` | `ModelChoice`, `ModelRouterMiddleware`, `AutoModeMiddleware`, `ModelRouterState` |

Middleware requires the `middleware` extra. `Noul` and `Score` are aliases whose
underlying schema names appear in the API reference. Public import aliases and
serialized class identifiers are compatibility commitments. Underscore-prefixed
modules, attributes, and methods are implementation details.

## Behavior applications can depend on

- `FinicClassifier` accepts a validated request or request dictionary and returns
  a `DecisionResponse`. `invoke`/`ainvoke`, LCEL composition, tool conversion,
  callbacks, configurable fields, and batching follow the documented Runnable usage.
- Batches make separate HTTP requests, preserve result order, and support
  `return_exceptions=True`. They are not atomic. Streaming emits one complete decision.
- Required fields, criteria matching, probability bounds/distributions, and model
  identity are validated. Additive provider response metadata is ignored when unused.
- The supported configuration is `FINIC_API_KEY`, `FINIC_MODEL`,
  `FINIC_TIMEOUT_SECONDS`, and `FINIC_MAX_ATTEMPTS`, plus the documented constructor
  options. The package does not read `.env`. `GRAYSON_*` aliases are not supported.
- Provider failures, exhausted transport retries, and response-contract violations
  raise `FinicError`, with optional `status` and `request_id`. Local Pydantic
  validation raises `ValidationError`; unsupported invocation keywords raise
  `TypeError`. Invalid lifecycle operations can raise `RuntimeError`. Exact error
  wording is not a stable API; use exception types and documented fields.
- Context managers reuse and close owned clients. Borrowed clients remain
  caller-owned; one event loop is supported per async client lifetime. Close after
  in-flight operations finish. Invocation-scoped clients close on completion or failure.
- Serializable configurations retain the model and a `FINIC_API_KEY` secret reference.
  Custom HTTP clients/transports are not serializable. Load only trusted data with
  the LangChain loader's required allowlist and supplied credentials.

The [usage guide](usage.md) defines defaults, precedence, retry behavior, lifecycle
details, tracing, and middleware fallback/blocking behavior. The
[API reference](api.md) defines field and method signatures. Finic controls model
availability and predictions; compatibility does not promise fixed probabilities,
calibration, latency, or application-specific detection accuracy.

## Python and dependency support

The package requires Python 3.11 or newer. CI currently tests Python 3.11, 3.12,
3.13, and 3.14 on Linux; newer Python versions and other platforms are not covered
by that matrix. Wheel installation is checked both with and without middleware.

Declared ranges are `langchain-core>=1,<2`, `langsmith>=0.3,<1`,
`pydantic>=2.12,<3`, and `httpx>=0.28.1,<1`. Middleware adds `langchain>=1,<2`
and `langgraph>=1,<2`. CI checks the locked development environment, installed
distributions, and an explicit minimum-dependency environment. This is not an
exhaustive test of every combination within these ranges.

If a supported combination fails, report the installed package versions and a
minimal reproduction with synthetic data. Changes to supported Python versions or
dependency lower bounds will be stated in release notes.

## Version and deprecation policy

During alpha and beta, proposed breaking changes require discussion and release
notes with migration instructions. Beta is intended for API validation and user
feedback; users should pin the tested prerelease in their application lockfile.

For the final `0.1.x` series, patch releases will preserve the documented public
imports, accepted call signatures, required request/response fields, serialized
class identifiers, and ownership behavior. Bug and security fixes may reject
previously accepted invalid input or unsafe behavior; release notes will explain
such changes. A `0.1.0` release is not a `1.0` stability guarantee.

Planned incompatible API changes will target a new minor series, such as `0.2.0`,
with migration instructions. Where practical, a warning and supported replacement
will be available in at least one published release before removal. Urgent security
fixes may shorten that notice, with an advisory explaining the impact.

Security fixes target the latest supported release as described in the repository's
[security policy](https://github.com/supreplabs/langchain-finic/blob/main/SECURITY.md).
There is no promised LTS period or response-time SLA.
