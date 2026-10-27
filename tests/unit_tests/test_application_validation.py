import asyncio
import json
import runpy
from pathlib import Path

import httpx
import pytest

from langchain_finic import FinicClassifier, FinicError

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("blocked", [False, True])
def test_application_consumers_and_request_budgets(blocked):
    module = runpy.run_path(str(ROOT / "scripts/validate_applications.py"))
    calls = []

    def handle(request):
        data = json.loads(request.content)
        calls.append(data)
        answers = {}
        for name, q in data["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.8 if blocked else 0.1}
            else:
                keys = list(q["criteria"])
                answers[name] = {
                    "type": "choice",
                    "choice": keys[0],
                    "confidence": 1,
                    "probabilities": {key: float(key == keys[0]) for key in keys},
                }
        return httpx.Response(
            200,
            json={
                "id": "demo",
                "model": data["model"],
                "answers": answers,
                "request_id": "req_demo",
                "usage": {"input_tokens": 1, "output_tokens": 0},
                "evaluation_time_ms": 1,
            },
        )

    async def run():
        async with FinicClassifier(
            api_key="fake", max_attempts=1, async_transport=httpx.MockTransport(handle)
        ) as classifier:
            for name, expected in [("payment_review", 2), ("checkpoint_review", 1), ("routed_review", 2)]:
                before = len(calls)
                result = await module[name](classifier)
                assert result and len(calls) - before == expected
        with pytest.raises(RuntimeError):
            await classifier.ainvoke(
                module["request"](
                    module["Payment"](
                        case_id="closed",
                        amount_inr=1,
                        account_age_days=1,
                        new_device=False,
                        usual_amount_inr=1,
                    )
                )
            )

    asyncio.run(run())
    assert len(calls) == 5


def test_failed_assessment_does_not_become_application_success():
    module = runpy.run_path(str(ROOT / "scripts/validate_applications.py"))

    async def run():
        async with FinicClassifier(
            api_key="fake", max_attempts=1, async_transport=httpx.MockTransport(lambda _: httpx.Response(503))
        ) as classifier:
            with pytest.raises(FinicError):
                await module["payment_review"](classifier)

    asyncio.run(run())


def test_application_runner_requires_explicit_opt_in(tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/validate_applications.py"))
    with pytest.raises(RuntimeError, match="opt-in"):
        asyncio.run(module["run"](tmp_path / "report.json", "0.1.0a6"))
