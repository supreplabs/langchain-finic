"""Finic accepts the same outcome vocabulary in JSON and CSV."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from langchain_finic import CalibrationRequest, CreateScoreRequest, FinicClient


@pytest.mark.parametrize(
    "bad,good", [("true", "false"), ("TRUE", "FALSE"), ("1", "0"), ("yes", "no"), ("YeS", "nO")]
)
@pytest.mark.parametrize("encoding", ["json", "csv"])
@pytest.mark.parametrize("operation", ["create_score", "calibrate"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_documented_label_forms_reach_wire_unchanged(bad, good, encoding, operation, asynchronous):
    labels = [bad if i % 2 else good for i in range(100)]
    if operation == "create_score":
        data = [{"amount": i, "is_bad": label} for i, label in enumerate(labels)]
        if encoding == "csv":
            data = "amount,is_bad\n" + "".join(f"{i},{label}\n" for i, label in enumerate(labels))
        request = CreateScoreRequest(data=data)
        result_body = {"model_id": "scm_test", "job_id": "job_test", "request_id": "req_test"}
    else:
        data = [{"score_id": f"s_{i}", "is_bad": label} for i, label in enumerate(labels)]
        if encoding == "csv":
            data = "score_id,is_bad\n" + "".join(f"s_{i},{label}\n" for i, label in enumerate(labels))
        request = CalibrationRequest(model_id="scm_test", data=data)
        result_body = {"job_id": "job_test", "request_id": "req_test", "matched": 100, "unmatched": 0}
    seen = []

    def handle(http_request):
        sent = json.loads(http_request.content)
        seen.append(sent)
        assert sent["data"] == data
        return httpx.Response(202, json=result_body)

    transport = httpx.MockTransport(handle)
    client = FinicClient("test", transport=transport, async_transport=transport)
    if asynchronous:

        async def run():
            async with client:
                return await getattr(client, "a" + operation)(request)

        result = asyncio.run(run())
    else:
        with client:
            result = getattr(client, operation)(request)
    assert result.job_id == "job_test" and len(seen) == 1
    assert request.data == data


@pytest.mark.parametrize("label", ["maybe", "", " true ", "2", 2, None, [], {}])
def test_invalid_labels_remain_rejected(label):
    with pytest.raises(ValidationError):
        CreateScoreRequest(data=[{"amount": i, "is_bad": label} for i in range(100)])
    with pytest.raises(ValidationError):
        CalibrationRequest(model_id="m", data=[{"score_id": "s", "is_bad": label}])


def test_text_labels_still_require_both_training_outcomes():
    with pytest.raises(ValidationError, match="20 examples of each outcome"):
        CreateScoreRequest(data=[{"amount": i, "is_bad": "yes"} for i in range(100)])
