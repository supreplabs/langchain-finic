"""Regressions for migration and tool capability feedback."""

import pytest
from pydantic import ValidationError

from langchain_finic import CalibrationRequest, FinicClient, FinicToolkit


@pytest.mark.parametrize("csv", [False, True])
def test_legacy_id_only_calibration_has_migration_diagnostic(csv):
    rows = [{"score_id": f"record-{i}", "is_bad": i % 2} for i in range(100)]
    data = "score_id,is_bad\n" + "\n".join(f"record-{i},{i % 2}" for i in range(100)) if csv else rows
    with pytest.raises(ValidationError, match="original feature records"):
        CalibrationRequest(model_id="model", data=data)


@pytest.mark.parametrize("flag", ["update_schema", "skip_schema_validation"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_toolkit_score_rejects_ungranted_schema_flags(flag, asynchronous):
    with FinicClient("test") as client:
        tool = FinicToolkit(client=client, operations=("score",)).get_tools()[0]
        payload = {"model_id": "model", "score_id": "record", "data": {"amount": 1}, flag: True}
        assert flag not in tool.args_schema.model_fields
        with pytest.raises(ValueError):
            if asynchronous:
                await tool.ainvoke(payload)
            else:
                tool.invoke(payload)


@pytest.mark.parametrize(
    "flag,option",
    [("update_schema", "allow_schema_update"), ("skip_schema_validation", "allow_schema_bypass")],
)
def test_toolkit_score_explicitly_grants_schema_capability(flag, option):
    import httpx

    seen = []

    def handle(request):
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model_id": "model", "score_id": "record", "score": 500, "request_id": "req"}
        )

    with FinicClient("test", transport=httpx.MockTransport(handle)) as client:
        tool = FinicToolkit(client=client, operations=("score",), **{option: True}).get_tools()[0]
        tool.invoke({"model_id": "model", "score_id": "record", "data": {"amount": 1}, flag: True})
    assert seen[0][flag] is True
