"""Offline control-flow checks for the application deadline example."""

import asyncio
import json
import runpy
from pathlib import Path

import httpx
import pytest

from langchain_finic import FinicClassifier

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/deadline_routing.py"
select_model = runpy.run_path(str(EXAMPLE))["select_model"]


def response(request, probability=0.9):
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "dec_deadline",
            "model": payload["model"],
            "answers": {
                "model_route": {
                    "type": "choice",
                    "choice": "fast",
                    "confidence": 2 * probability - 1,
                    "probabilities": {"fast": probability, "powerful": 1 - probability},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "request_id": "req_deadline",
            "evaluation_time_ms": 1,
        },
    )


@pytest.mark.parametrize(
    "probability,selected,reason", [(0.9, "fast", "decision"), (0.6, "powerful", "low_probability")]
)
def test_decision_and_low_probability(probability, selected, reason):
    async def run():
        calls = []

        async def handle(request):
            calls.append(request)
            return response(request, probability)

        async with FinicClassifier(api_key="test", async_transport=httpx.MockTransport(handle)) as classifier:
            result = await select_model(classifier, "Fictional example")
            assert result.selected == selected and result.reason == reason
            assert len(calls) == 1

    asyncio.run(run())


def test_deadline_cancels_pending_http_without_replay():
    async def run():
        calls = 0
        cancelled = asyncio.Event()

        async def handle(request):
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async with FinicClassifier(api_key="test", async_transport=httpx.MockTransport(handle)) as classifier:
            result = await select_model(classifier, "Fictional example", deadline_seconds=0.01)
            assert result.selected == "powerful" and result.reason == "deadline"
            assert cancelled.is_set() and calls == 1

    asyncio.run(run())


def test_caller_cancellation_propagates():
    async def run():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handle(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async with FinicClassifier(api_key="test", async_transport=httpx.MockTransport(handle)) as classifier:
            task = asyncio.create_task(select_model(classifier, "Fictional example", deadline_seconds=10))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()

    asyncio.run(run())


def test_provider_failure_uses_fallback_without_retry():
    async def run():
        calls = 0

        async def handle(request):
            nonlocal calls
            calls += 1
            return httpx.Response(503)

        async with FinicClassifier(
            api_key="test", max_attempts=1, async_transport=httpx.MockTransport(handle)
        ) as classifier:
            result = await select_model(classifier, "Fictional example")
            assert result.selected == "powerful" and result.reason == "provider_error"
            assert calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("deadline", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_fails_before_http(deadline):
    async def run():
        def unexpected(request):
            pytest.fail("Invalid deadline must not send a request")

        async with FinicClassifier(
            api_key="test", async_transport=httpx.MockTransport(unexpected)
        ) as classifier:
            with pytest.raises(ValueError, match="positive finite"):
                await select_model(classifier, "Fictional example", deadline_seconds=deadline)

    asyncio.run(run())


def test_repeated_decisions_reuse_and_close_the_entered_pool():
    async def run():
        async with FinicClassifier(
            api_key="test", async_transport=httpx.MockTransport(response)
        ) as classifier:
            await select_model(classifier, "First fictional query")
            pool = classifier._client._async_client
            await select_model(classifier, "Second fictional query")
            assert classifier._client._async_client is pool
            assert pool is not None and not pool.is_closed
        assert pool.is_closed

    asyncio.run(run())
