"""Finic accepts the same outcome vocabulary in JSON and CSV."""

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from langchain_finic import CalibrationRequest, CreateScoringModelRequest, FinicClient


@pytest.mark.parametrize(
    "bad,good", [("true", "false"), ("TRUE", "FALSE"), ("1", "0"), ("yes", "no"), ("YeS", "nO")]
)
@pytest.mark.parametrize("encoding", ["json", "csv"])
@pytest.mark.parametrize("operation", ["create_scoring_model", "calibrate"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_documented_label_forms_reach_wire_unchanged(bad, good, encoding, operation, asynchronous):
    labels = [bad if i % 2 else good for i in range(100)]
    data = [{"amount": i, "is_bad": label} for i, label in enumerate(labels)]
    if encoding == "csv":
        data = "amount,is_bad\n" + "".join(f"{i},{label}\n" for i, label in enumerate(labels))
    if operation == "create_scoring_model":
        request = CreateScoringModelRequest(name="Labels", data=data)
        result_body = {"model_id": "scm_test", "job_id": "job_test", "request_id": "req_test"}
    else:
        request = CalibrationRequest(model_id="scm_test", data=data)
        result_body = {"job_id": "job_test", "request_id": "req_test", "rows": 100}
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
    rows = [{"amount": i, "is_bad": label} for i in range(100)]
    with pytest.raises(ValidationError):
        CreateScoringModelRequest(name="M", data=rows)
    with pytest.raises(ValidationError):
        CalibrationRequest(model_id="m", data=rows)


def test_text_labels_still_require_both_outcomes():
    rows = [{"amount": i, "is_bad": "yes"} for i in range(100)]
    with pytest.raises(ValidationError, match="20 examples of each outcome"):
        CreateScoringModelRequest(name="M", data=rows)
    with pytest.raises(ValidationError, match="20 examples of each outcome"):
        CalibrationRequest(model_id="m", data=rows)
