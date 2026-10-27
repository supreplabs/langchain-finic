"""Failure contracts observable by applications; no provider traffic or real sleeps."""

import asyncio
import json
from collections import Counter

import httpx
import pytest

from langchain_finic import FinicClassifier, FinicError


class ClosingTransport(httpx.MockTransport):
    closed = False

    def close(self):
        self.closed = True
        super().close()

    async def aclose(self):
        self.closed = True
        await super().aclose()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("failure", ["timeout", "rate_limit", "server", "invalid_json"])
def test_failure_budget_redaction_and_cleanup(monkeypatch, request_body, async_mode, failure):
    monkeypatch.setattr("langchain_finic.client._delay", lambda _: 0)
    calls = []
    status = {"rate_limit": 429, "server": 503, "invalid_json": 200}.get(failure)

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private-network-and-case-details", request=request)
        return httpx.Response(status, text="private-case-details", headers={"x-request-id": "req_failed"})

    transport = ClosingTransport(handle)

    async def run():
        async with FinicClassifier(api_key="fake", max_attempts=2, async_transport=transport) as model:
            await model.ainvoke(request_body)

    with pytest.raises(FinicError) as caught:
        if async_mode:
            asyncio.run(run())
        else:
            with FinicClassifier(api_key="fake", max_attempts=2, transport=transport) as model:
                model.invoke(request_body)
    assert len(calls) == (1 if failure == "invalid_json" else 2)
    assert transport.closed
    assert "private" not in str(caught.value)
    assert caught.value.status == (status if failure in {"rate_limit", "server"} else None)
    assert caught.value.request_id == (None if failure == "timeout" else "req_failed")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("return_exceptions", [False, True])
def test_partial_batch_preserves_order_or_raises(request_body, response_body, async_mode, return_exceptions):
    calls = []

    def handle(request):
        case = json.loads(request.content)["data"]["case"]
        calls.append(case)
        if case == "bad":
            return httpx.Response(401, text="private-case-details")
        return httpx.Response(200, json={**response_body, "request_id": case})

    transport = ClosingTransport(handle)
    inputs = [request_body.model_copy(update={"state": {"case": case}}) for case in ("first", "bad", "last")]
    config = {"max_concurrency": 1}

    async def run():
        async with FinicClassifier(api_key="fake", max_attempts=1, async_transport=transport) as model:
            return await model.abatch(inputs, config, return_exceptions=return_exceptions)

    def invoke():
        if async_mode:
            return asyncio.run(run())
        with FinicClassifier(api_key="fake", max_attempts=1, transport=transport) as model:
            return model.batch(inputs, config, return_exceptions=return_exceptions)

    if return_exceptions:
        results = invoke()
        assert len(results) == 3
        assert results[0].request_id == "first" and results[2].request_id == "last"
        assert isinstance(results[1], FinicError) and results[1].status == 401
        assert Counter(calls) == {"first": 1, "bad": 1, "last": 1}
    else:
        with pytest.raises(FinicError) as caught:
            invoke()
        assert caught.value.status == 401
        assert Counter(calls)["bad"] == 1
        assert all(count == 1 for count in Counter(calls).values())
    assert transport.closed


def test_cancellation_during_backoff_stops_requests_and_closes(request_body, monkeypatch):
    calls = []
    waiting = asyncio.Event()

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    async def backoff(_):
        waiting.set()
        await asyncio.Future()

    monkeypatch.setattr("langchain_finic.client.asyncio.sleep", backoff)
    transport = ClosingTransport(handle)

    async def run():
        async with FinicClassifier(api_key="fake", async_transport=transport) as model:
            task = asyncio.create_task(model.ainvoke(request_body))
            await asyncio.wait_for(waiting.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
    assert len(calls) == 1
    assert transport.closed
