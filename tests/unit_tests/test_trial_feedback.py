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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_toolkit_score_normalizes_langchain_messages(asynchronous):
    import json

    import httpx
    from langchain_core.messages import HumanMessage

    from langchain_finic import ScoringRequest

    payload = {
        "model_id": "model",
        "score_id": "record",
        "data": {"messages": [HumanMessage(content="Review this fictional transaction")]},
    }
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model_id": "model", "score_id": "record", "score": 500, "request_id": "req"}
        )

    transport = httpx.MockTransport(handle)
    async with FinicClient("test", transport=transport, async_transport=transport) as client:
        tool = FinicToolkit(client=client, operations=("score",)).get_tools()[0]
        if asynchronous:
            await tool.ainvoke(payload)
        else:
            tool.invoke(payload)
    assert seen == [ScoringRequest(**payload).model_dump(exclude_none=True)]
    assert seen[0]["data"] == {"messages": [{"role": "user", "content": "Review this fictional transaction"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_both_schema_permissions_are_forwarded_and_revocation_blocks_old_tool(asynchronous):
    import json

    import httpx

    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model_id": "model", "score_id": "record", "score": 500, "request_id": "req"}
        )

    transport = httpx.MockTransport(handle)
    async with FinicClient("test", transport=transport, async_transport=transport) as client:
        toolkit = FinicToolkit(
            client=client, operations=("score",), allow_schema_update=True, allow_schema_bypass=True
        )
        tool = toolkit.get_tools()[0]
        payload = {
            "model_id": "model",
            "score_id": "record",
            "data": {"amount": 1},
            "update_schema": True,
            "skip_schema_validation": True,
        }
        invoke = tool.ainvoke if asynchronous else tool.invoke
        if asynchronous:
            await invoke(payload)
        else:
            invoke(payload)
        assert seen == [payload]
        toolkit.allow_schema_update = False
        with pytest.raises(ValueError, match="explicit toolkit opt-in"):
            if asynchronous:
                await invoke(payload)
            else:
                invoke(payload)
        assert seen == [payload]
        assert "update_schema" not in toolkit.get_tools()[0].args_schema.model_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "data", [{}, [{"amount": 1}, {"amount": 2}], {"amount": float("nan")}, "amount\n1\n2\n"]
)
async def test_score_tool_rejects_malformed_records_before_http(asynchronous, data):
    import httpx

    seen = []

    def handle(request):
        seen.append(request)
        raise AssertionError("Invalid records must not reach HTTP")

    transport = httpx.MockTransport(handle)
    async with FinicClient("test", transport=transport, async_transport=transport) as client:
        tool = FinicToolkit(client=client, operations=("score",)).get_tools()[0]
        payload = {"model_id": "model", "score_id": "record", "data": data}
        with pytest.raises(ValueError):
            if asynchronous:
                await tool.ainvoke(payload)
            else:
                tool.invoke(payload)
    assert not seen


def test_new_permission_requires_rebuilding_tool_schema():
    with FinicClient("test") as client:
        toolkit = FinicToolkit(client=client, operations=("score",))
        old_tool = toolkit.get_tools()[0]
        toolkit.allow_schema_bypass = True
        assert "skip_schema_validation" not in old_tool.args_schema.model_fields
        assert "skip_schema_validation" in toolkit.get_tools()[0].args_schema.model_fields
        with pytest.raises(ValueError):
            old_tool.invoke(
                {"model_id": "m", "score_id": "s", "data": {"amount": 1}, "skip_schema_validation": True}
            )
