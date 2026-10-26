"""Explicit opt-in: exactly two paid calls, no retries or customer data."""

import asyncio
import os

import pytest

from langchain_finic import DecisionRequest, FinicClassifier, Noul

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("FINIC_LIVE_TESTS") != "1" or not os.getenv("FINIC_API_KEY"),
        reason="Set FINIC_LIVE_TESTS=1 and export FINIC_API_KEY to spend API credits",
    ),
]


def request():
    return DecisionRequest(
        state={"events": ["Fictional customer uses their usual device for a routine payment"]},
        questions={"risk": Noul(instructions="Does the evidence suggest fraud?")},
    )


def test_live_sync():
    with FinicClassifier(max_attempts=1) as classifier:
        result = classifier.invoke(request())
    assert 0 <= result.nouls["risk"].noul <= 1
    assert result.request_id


def test_live_async():
    async def run():
        async with FinicClassifier(max_attempts=1) as classifier:
            return await classifier.ainvoke(request())

    result = asyncio.run(run())
    assert 0 <= result.nouls["risk"].noul <= 1
    assert result.request_id
