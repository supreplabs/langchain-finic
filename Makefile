.PHONY: install test integration_test lifecycle_test lint format type_check docs build check
install:
	uv sync --locked --group dev --extra middleware
test:
	uv run --locked --group test pytest tests/unit_tests -q
integration_test:
	FINIC_LIVE_TESTS=1 uv run --locked --group test python -m pytest -o addopts= tests/integration_tests -m live -q $(LIVE_TEST_ARGS)
lifecycle_test:
	uv run --locked --group test pytest -o addopts= tests/integration_tests -m lifecycle -q
lint:
	uv run --locked --group lint ruff check src tests examples scripts
	uv run --locked --group lint ruff format --check src tests examples scripts
format:
	uv run --locked --group lint ruff format src tests examples scripts
type_check:
	uv run --locked --group lint --extra middleware mypy
docs:
	uv run --locked --group docs --extra middleware mkdocs build --strict
build:
	uv build
	uv run --locked --group build twine check dist/*
	uv run --locked python scripts/check_dist.py
check: lint type_check test docs build
