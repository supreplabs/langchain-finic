# Contributing

Use Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/supreplabs/langchain-finic.git
cd langchain-finic
make install
make check
```

Keep changes focused, describe the behavior and validation in your pull request,
and add regression tests for bugs. Public imports and serialized class identifiers
are compatibility commitments, including during the alpha period. Discuss breaking
changes in an issue before implementing them. Use conventional commit titles.

`tests/unit_tests` includes mocked framework composition tests and LangChain's
standard tool tests. Network access is disabled. Tests must not load `.env` or
require credentials. `langchain-tests` is pinned because its test contracts can
change independently of this package. Update it deliberately with the lockfile.

`make integration_test` runs LangChain's official `ToolsIntegrationTests` against
the live Finic API, plus our sync and async classifier smoke tests. With the pinned
`langchain-tests` version, this runs nine tests and makes eight paid API requests:
four tool calls (sync and async, with and without a `ToolCall`) and two classifier
calls, plus two documented data/context checks. Each request gets one attempt, using synthetic evidence. The remaining test
checks that the inherited standard tests have not been overridden.

Export `FINIC_API_KEY` before running the command; the Makefile sets
`FINIC_LIVE_TESTS=1` and enables network access for this opt-in run. Direct pytest
runs require both environment variables and `-o addopts=` to enable network access.
The live suite is excluded from default tests and pull-request CI, and skips unless both
environment variables are set. Never put keys, production inputs, or raw responses
in issues or fixtures. These tests check response contracts, not deterministic
probability values.

The **Live integration tests** GitHub Actions workflow runs daily on `main` at
03:17 UTC and can be triggered manually from the Actions tab on `main`. Future
release workflows also run it before building and publishing. Configure the
repository secret `FINIC_API_KEY` to enable these runs. If it is absent, the live
job is skipped with an explanation in the workflow summary; releases can proceed.
If it is present, live test failures fail the workflow and block publishing.
Each enabled run uses Python 3.12 and makes eight API requests without retries.

Use `make format` for formatting and `uv lock` after dependency changes.
Documentation lives in `docs/`; public API pages are generated from docstrings.
See [release instructions](RELEASING.md) for distribution and publishing.

By contributing, you agree to license your contributions under this repository's
MIT license. See [community conduct](CODE_OF_CONDUCT.md) and
[security reporting](SECURITY.md).

The minimum-dependency CI job uses `langchain-tests==1.0.0`, which supports
`langchain-core==1.0.0`; current-dependency jobs use `langchain-tests==1.1.9`.
The latter requires core >=1.4 and cannot validate the historical minimum.

## Documentation standards

Keep the user site focused on installation, supported interfaces, configuration,
examples, and API behavior. Release procedures and directory-listing instructions
belong in the repository's maintainer guides, outside `docs/`.

Ground capability claims in the public API and tests. Do not publish internal
research results, account-specific usage, or unverified accuracy/calibration claims.
Use fictional examples. Define inputs explicitly or label examples as reusable
functions with caller-supplied parameters. Python snippets in the README and user
guides are executed offline by `test_documentation.py`; keep them runnable in the
order shown. Prefer links over duplicated explanations and check relative links.

Follow [LangChain's documentation quality guidance](https://docs.langchain.com/oss/python/contributing/documentation#quality-standards).
Its Mintlify-specific syntax applies to LangChain-hosted pages; this repository's
site uses MkDocs and generated Python API references.
