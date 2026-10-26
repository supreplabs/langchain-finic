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

`make integration_test` makes two paid API requests (sync and async), with one
attempt each, using synthetic evidence. It requires an exported `FINIC_API_KEY`;
never put keys, production inputs, or raw responses in issues or fixtures.
These tests check response contracts, not deterministic probability values.

Use `make format` for formatting and `uv lock` after dependency changes.
Documentation lives in `docs/`; public API pages are generated from docstrings.
See [release instructions](RELEASING.md) for distribution and publishing.

By contributing, you agree to license your contributions under this repository's
MIT license. See [community conduct](CODE_OF_CONDUCT.md) and
[security reporting](SECURITY.md).

The minimum-dependency CI job uses `langchain-tests==1.0.0`, which supports
`langchain-core==1.0.0`; current-dependency jobs use `langchain-tests==1.1.9`.
The latter requires core >=1.4 and cannot validate the historical minimum.
