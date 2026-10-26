"""Public LangChain contracts and resource ownership; no paid services required."""

import asyncio
import inspect
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import ConfigurableField
from pydantic import ValidationError

from langchain_finic import DecisionRequest, FinicClassifier, FinicClient, Noul


@pytest.mark.parametrize("fails", [False, True])
def test_configurable_sync_pools_closed_per_invocation(monkeypatch, request_body, response_body, fails):
    pools = []
    original = httpx.Client

    def handle(request):
        assert request.extensions["timeout"]["read"] == 75
        return httpx.Response(400 if fails else 200, json=response_body)

    def tracked(**kwargs):
        pool = original(**kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(httpx, "Client", tracked)
    # Borrowed transport avoids making pool construction depend on real networking.
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handle)) as model:
        dynamic = model.configurable_fields(timeout=ConfigurableField(id="request_timeout"))
        configured = dynamic.with_config(configurable={"request_timeout": 75})
        if fails:
            from langchain_finic import FinicError

            for _ in range(2):
                with pytest.raises(FinicError):
                    configured.invoke(request_body)
        else:
            assert len(configured.batch([request_body] * 4, {"max_concurrency": 2})) == 4
        assert len(pools) == (2 if fails else 4)
        assert all(pool.is_closed for pool in pools)


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_configurable_async_pools_closed(monkeypatch, request_body, response_body, outcome):
    from langchain_finic import FinicError

    pools = []
    original = httpx.AsyncClient
    started = asyncio.Event()

    async def handle(request):
        assert request.extensions["timeout"]["read"] == 75
        if outcome == "cancel":
            started.set()
            await asyncio.Future()
        return httpx.Response(400 if outcome == "failure" else 200, json=response_body)

    def tracked(**kwargs):
        pool = original(**kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(httpx, "AsyncClient", tracked)

    async def run():
        async with FinicClassifier(api_key="fake", async_transport=httpx.MockTransport(handle)) as model:
            dynamic = model.configurable_fields(timeout=ConfigurableField(id="request_timeout"))
            configured = dynamic.with_config(configurable={"request_timeout": 75})
            if outcome == "success":
                assert len(await configured.abatch([request_body] * 4)) == 4
            else:
                for _ in range(2):
                    if outcome == "cancel":
                        started.clear()
                        task = asyncio.create_task(configured.ainvoke(request_body))
                        await asyncio.wait_for(started.wait(), timeout=5)
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    else:
                        with pytest.raises(FinicError):
                            await configured.ainvoke(request_body)
            assert len(pools) == (4 if outcome == "success" else 2)
            assert all(pool.is_closed for pool in pools)

    asyncio.run(run())


def test_configured_models_leave_borrowed_http_pools_open(request_body, handler):
    with httpx.Client(transport=httpx.MockTransport(handler)) as pool:
        with FinicClassifier(api_key="fake", http_client=pool) as model:
            configured = model.configurable_fields(timeout=ConfigurableField(id="request_timeout"))
            assert len(configured.batch([request_body] * 2, {"configurable": {"request_timeout": 75}})) == 2
        assert not pool.is_closed

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as pool:
            async with FinicClassifier(api_key="fake", http_async_client=pool) as model:
                configured = model.configurable_fields(timeout=ConfigurableField(id="request_timeout"))
                responses = await configured.abatch(
                    [request_body] * 2, {"configurable": {"request_timeout": 75}}
                )
                assert len(responses) == 2
            assert not pool.is_closed

    asyncio.run(run())


def test_typed_constructor_and_bound_invocation(request_body, handler):
    assert {"api_key", "model", "timeout", "max_attempts"} <= set(
        inspect.signature(FinicClassifier).parameters
    )
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as model:
        assert model.bind(model="grayson-1").invoke(request_body).id == "dec_test"
        with pytest.raises(ValidationError):
            model.bind(model="invalid model").invoke(request_body)
        with pytest.raises(TypeError, match="temperature"):
            model.bind(temperature=0).invoke(request_body)
        with pytest.raises(ValidationError):
            FinicClassifier(api_key="fake", unknown_setting=True)


def test_serialization_redacts_and_reloads_explicit_secret():
    with FinicClassifier(api_key="do-not-persist-this") as model:
        serialized = dumps(model)
        assert "do-not-persist-this" not in serialized + repr(model) + model.model_dump_json()
        assert '"type": "secret"' in serialized
        options = (
            {"allowed_objects": [FinicClassifier]}
            if "allowed_objects" in inspect.signature(loads).parameters
            else {}
        )
        with loads(
            serialized,
            secrets_map={"FINIC_API_KEY": "replacement"},
            valid_namespaces=["langchain_finic"],
            **options,
        ) as restored:
            assert restored.api_key.get_secret_value() == "replacement"
            assert restored.timeout == 60


def test_custom_transport_not_silently_serialized(handler):
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as model:
        assert model.to_json()["type"] == "not_implemented"


def test_configurable_fields_reconstruct_validated_model(request_body, handler):
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as model:
        dynamic = model.configurable_fields(timeout=ConfigurableField(id="request_timeout"))
        assert dynamic.with_config(configurable={"request_timeout": 75}).invoke(request_body).id == "dec_test"
        with pytest.raises(ValidationError):
            dynamic.with_config(configurable={"request_timeout": 0}).invoke(request_body)


@pytest.mark.parametrize(
    "name,legacy,preferred,field",
    [
        ("TIMEOUT_SECONDS", "65", "75", "timeout"),
        ("MAX_ATTEMPTS", "2", "4", "max_attempts"),
    ],
)
def test_client_environment_options(monkeypatch, name, legacy, preferred, field):
    monkeypatch.setenv(f"FINIC_{name}", legacy)
    monkeypatch.setenv(f"GRAYSON_{name}", preferred)
    with FinicClassifier(api_key="fake") as model:
        assert getattr(model, field) == float(legacy)
    monkeypatch.delenv(f"GRAYSON_{name}")
    with FinicClassifier(api_key="fake") as model:
        assert getattr(model, field) == float(legacy)


def test_messages_preserve_tool_protocol_without_mutation():
    messages = [
        HumanMessage(content="Review", id="private-framework-id"),
        AIMessage(content="", tool_calls=[{"id": "call1", "name": "lookup", "args": {"id": 7}}]),
        ToolMessage(content='{"status":"verified"}', tool_call_id="call1"),
    ]
    request = DecisionRequest(state={"messages": messages}, questions={"risk": Noul(instructions="Risk?")})
    wire = request.model_dump_json()
    assert "private-framework-id" not in wire
    assert request.state["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert request.state["messages"][2]["tool_call_id"] == "call1"
    assert isinstance(messages[0], HumanMessage)
    assert DecisionRequest(state=messages[0], questions=request.questions).state["role"] == "user"


@pytest.mark.parametrize("state", [None, True, 42, {1: "value"}, {"value": float("nan")}, b"bytes", object()])
def test_invalid_state_rejected(state):
    with pytest.raises(ValidationError):
        DecisionRequest(state=state, questions={"risk": Noul(instructions="Risk?")})


def test_trace_metadata_usage_and_accessors(request_body, handler, monkeypatch):
    trace = SimpleNamespace(extra={})
    monkeypatch.setattr("langchain_finic.classifier.get_current_run_tree", lambda: trace)

    class Recorder(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, **kwargs):
            self.start = kwargs

    recorder = Recorder()
    config = {"callbacks": [recorder], "tags": ["benchmark"], "metadata": {"experiment": "baseline"}}
    with FinicClassifier(api_key="private", transport=httpx.MockTransport(handler)) as model:
        result = model.invoke(request_body, config)
    assert recorder.start["metadata"]["ls_provider"] == "finic"
    assert recorder.start["metadata"]["ls_model_name"] == "grayson-1"
    assert recorder.start["metadata"]["experiment"] == "baseline"
    assert recorder.start["run_type"] == "llm"
    assert config["metadata"] == {"experiment": "baseline"}
    assert trace.extra["metadata"]["usage_metadata"] == {
        "input_tokens": 100,
        "output_tokens": 0,
        "total_tokens": 100,
    }
    assert result.nouls["fits_profile"].noul == 0.2
    assert result.choices["action"].choice == "escalate"
    assert result.multi_choices["indicators"].multi_choice == ["device", "payee"]
    assert result.scores["risk"].score == 1.6


def test_full_tool_signature(request_body, handler):
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as model:
        tool = model.as_tool(DecisionRequest, name="review", description="Review evidence")
        assert tool.name == "review"
        assert tool.invoke(request_body.model_dump())["id"] == "dec_test"
        tool2 = model.as_tool(arg_types={"state": str, "questions": dict})
        assert (
            tool2.invoke({"state": "evidence", "questions": request_body.model_dump()["questions"]})["id"]
            == "dec_test"
        )


def test_owned_pool_reused_closed_and_cannot_reopen(request_body, handler):
    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as model:
        model.invoke(request_body)
        pool = model._client._sync_client
        model.batch([request_body] * 4, {"max_concurrency": 2})
        assert pool is model._client._sync_client
    assert pool.is_closed
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.invoke(request_body)


def test_injected_client_ownership_and_headers(request_body, response_body):
    def handler(request):
        assert request.headers["authorization"] == "Bearer correct"
        return httpx.Response(200, json=response_body)

    with httpx.Client(transport=httpx.MockTransport(handler), headers={"Authorization": "wrong"}) as pool:
        with FinicClient("correct", http_client=pool) as client:
            with FinicClassifier(client=client) as model:
                model.invoke(request_body)
            assert client.decide(request_body).id == "dec_test"
        assert not pool.is_closed


def test_borrowed_client_cannot_follow_auth_redirect(request_body):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(307, headers={"location": "https://elsewhere.invalid"})

    with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as pool:
        with FinicClient("fake", http_client=pool) as client:
            with pytest.raises(Exception, match="HTTP 307"):
                client.decide(request_body)
    assert len(calls) == 1


def test_async_pool_and_bound_tool_lifecycle(request_body, handler):
    async def run():
        async with FinicClassifier(api_key="fake", async_transport=httpx.MockTransport(handler)) as model:
            assert (await model.bind(model="grayson-1").ainvoke(request_body)).id == "dec_test"
            pool = model._client._async_client
            assert len(await model.abatch([request_body] * 3)) == 3
            assert model._client._async_client is pool
            assert (await model.as_tool().ainvoke(request_body.model_dump()))["id"] == "dec_test"
            with pytest.raises(RuntimeError, match="aclose"):
                model.close()
        assert pool.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await model.ainvoke(request_body)

    asyncio.run(run())


def test_cancellation_does_not_retry(request_body):
    calls = []

    async def handle(request):
        calls.append(request)
        raise asyncio.CancelledError()

    async def run():
        async with FinicClassifier(api_key="fake", async_transport=httpx.MockTransport(handle)) as model:
            with pytest.raises(asyncio.CancelledError):
                await model.ainvoke(request_body)

    asyncio.run(run())
    assert len(calls) == 1


def test_langgraph_tool_node_and_checkpoint(request_body, handler):
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as model:
        tool = model.as_tool()
        graph = (
            StateGraph(MessagesState)
            .add_node("tools", ToolNode([tool]))
            .add_edge(START, "tools")
            .add_edge("tools", END)
            .compile(checkpointer=InMemorySaver())
        )
        config = {"configurable": {"thread_id": "decision-test"}}
        result = graph.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": tool.name,
                                "args": request_body.model_dump(mode="json"),
                            }
                        ],
                    )
                ]
            },
            config,
        )
        assert json.loads(result["messages"][-1].content)["request_id"] == "req_test"
        assert graph.get_state(config).values["messages"][-1].tool_call_id == "call_1"


def test_create_agent_calls_finic_tool_sync_and_async(request_body, handler):
    pytest.importorskip("langchain")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def agent_for(classifier):
        tool = classifier.as_tool()
        chat = ToolCallingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "decision_1",
                            "name": tool.name,
                            "args": request_body.model_dump(mode="json"),
                        }
                    ],
                ),
                AIMessage(content="Send to investigator review."),
            ]
        )
        return create_agent(chat, tools=[tool])

    with FinicClassifier(api_key="fake", transport=httpx.MockTransport(handler)) as classifier:
        result = agent_for(classifier).invoke({"messages": [HumanMessage(content="Review this alert")]})
        assert result["messages"][-1].content == "Send to investigator review."
        assert json.loads(result["messages"][-2].content)["answers"]["action"]["choice"] == "escalate"

    async def run():
        async with FinicClassifier(
            api_key="fake", async_transport=httpx.MockTransport(handler)
        ) as classifier:
            result = await agent_for(classifier).ainvoke({"messages": [HumanMessage(content="Review")]})
            assert json.loads(result["messages"][-2].content)["id"] == "dec_test"

    asyncio.run(run())


def test_config_schema_omits_runtime_handles():
    properties = FinicClassifier.model_json_schema()["properties"]
    assert {"model", "timeout", "max_attempts"} <= properties.keys()
    assert not {"client", "http_client", "transport", "async_transport"} & properties.keys()
