import asyncio
import inspect
import json
import runpy
from pathlib import Path

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.load import dumps, loads
from langchain_core.runnables import ConfigurableField
from langchain_tests.integration_tests.tools import ToolsIntegrationTests
from langchain_tests.unit_tests.tools import ToolsUnitTests
from pydantic import ValidationError

from langchain_finic import FinicClient, FinicError, FinicScorer, FinicToolkit, ScoringInput


def score_response(request):
    data = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "model_id": data["model_id"],
            "score_id": data["score_id"],
            "score": 200,
            "request_id": "req_score",
        },
    )


def scorer():
    return FinicScorer(
        model_id="scm_test",
        api_key="test",
        transport=httpx.MockTransport(score_response),
        async_transport=httpx.MockTransport(score_response),
    )


def record(i=0):
    return {"score_id": f"record_{i}", "data": {"amount": i}}


class ScorerContract:
    @property
    def tool_constructor(self):
        return scorer().as_tool()

    @property
    def tool_invoke_params_example(self):
        return record()


class TestScorerUnit(ScorerContract, ToolsUnitTests):
    pass


class TestScorerIntegration(ScorerContract, ToolsIntegrationTests):
    pass


class ToolkitContract:
    @pytest.fixture(autouse=True, params=["list_scores", "get_job_status", "create_score", "calibrate"])
    def selected_operation(self, request):
        self.operation = request.param
        self.clients = []
        yield
        for client in self.clients:
            asyncio.run(client.aclose())

    @property
    def tool_constructor(self):
        responses = {
            "list_scores": {"scores": []},
            "get_job_status": {"job_id": "job_test", "complete": True},
            "create_score": {"model_id": "scm_test", "job_id": "job_test", "request_id": "req_test"},
            "calibrate": {"job_id": "job_test", "matched": 1, "unmatched": 0, "request_id": "req_test"},
        }
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=responses[self.operation]))
        client = FinicClient("test", transport=transport, async_transport=transport)
        self.clients.append(client)
        return FinicToolkit(client=client, operations=(self.operation,)).get_tools()[0]

    @property
    def tool_invoke_params_example(self):
        return {
            "list_scores": {},
            "get_job_status": {"job_id": "job_test"},
            "create_score": {"data": [{"amount": i, "is_bad": i % 2} for i in range(100)]},
            "calibrate": {"model_id": "scm_test", "data": [{"score_id": "s", "is_bad": False}]},
        }[self.operation]


class TestToolkitUnit(ToolkitContract, ToolsUnitTests):
    pass


class TestToolkitIntegration(ToolkitContract, ToolsIntegrationTests):
    pass


def test_lcel_callbacks_schemas_and_context():
    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.metadata = []

        def on_chain_start(self, serialized, inputs, **kwargs):
            self.metadata.append(kwargs.get("metadata", {}))

    callback = Recorder()
    with scorer() as model:
        assert model.get_input_schema() is ScoringInput
        assert (model | (lambda result: result.score)).invoke(record(), {"callbacks": [callback]}) == 200
        assert model._client._sync_client is not None
        assert model._client._sync_client.is_closed is False
        assert [r.score_id for r in model.batch([record(i) for i in range(3)], {"max_concurrency": 1})] == [
            "record_0",
            "record_1",
            "record_2",
        ]
        assert list(model.stream(record()))[0].score == 200
    assert model._client._sync_client.is_closed
    assert any(
        m.get("finic_api") == "score" and m.get("finic_model_id") == "scm_test" for m in callback.metadata
    )
    with pytest.raises(RuntimeError, match="closed"):
        model.invoke(record())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_partial_batch_preserves_ids_and_errors(asynchronous):
    def handle(request):
        body = json.loads(request.content)
        if body["score_id"] == "record_1":
            return httpx.Response(409, json={"error": {"type": "duplicate_score_id"}})
        return score_response(request)

    transport = httpx.MockTransport(handle)
    model = FinicScorer(model_id="scm_test", api_key="test", transport=transport, async_transport=transport)
    if asynchronous:

        async def run():
            async with model:
                return await model.abatch(
                    [record(i) for i in range(3)], {"max_concurrency": 2}, return_exceptions=True
                )

        results = asyncio.run(run())
    else:
        with model:
            results = model.batch(
                [record(i) for i in range(3)], {"max_concurrency": 2}, return_exceptions=True
            )
    assert results[0].score_id == "record_0" and results[2].score_id == "record_2"
    assert isinstance(results[1], FinicError) and results[1].error_type == "duplicate_score_id"


@pytest.mark.parametrize("limit", [None, 2])
def test_async_batch_concurrency(limit):
    async def run():
        active = peak = 0

        async def handle(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.005)
            active -= 1
            return score_response(request)

        async with FinicScorer(
            model_id="scm_test", api_key="test", async_transport=httpx.MockTransport(handle)
        ) as model:
            config = None if limit is None else {"max_concurrency": limit}
            assert len(await model.abatch([record(i) for i in range(12)], config)) == 12
            assert peak == (limit or 5)

    asyncio.run(run())


def test_serialization_and_configurable_model():
    model = FinicScorer(model_id="scm_test", api_key="private-key")
    payload = dumps(model)
    assert "private-key" not in payload and "FINIC_API_KEY" in payload
    options = (
        {"allowed_objects": [FinicScorer]} if "allowed_objects" in inspect.signature(loads).parameters else {}
    )
    with loads(
        payload,
        secrets_map={"FINIC_API_KEY": "restored-key"},
        valid_namespaces=["langchain_finic"],
        **options,
    ) as restored:
        assert restored.model_id == "scm_test"
        assert restored.api_key.get_secret_value() == "restored-key"
    model.close()
    with scorer() as borrowed:
        assert borrowed.to_json()["type"] == "not_implemented"
        configurable = borrowed.configurable_fields(model_id=ConfigurableField(id="score_model"))
        assert (
            configurable.invoke(record(), {"configurable": {"score_model": "scm_other"}}).model_id
            == "scm_other"
        )


def test_borrowed_client_tool_selection_and_validation():
    with FinicClient("test", transport=httpx.MockTransport(score_response)) as client:
        model = FinicScorer(model_id="scm_test", client=client)
        model.invoke(record())
        model.close()
        assert not client._closed
        toolkit = FinicToolkit(client=client)
        assert [tool.name for tool in toolkit.get_tools()] == ["finic_list_scores", "finic_get_job_status"]
        assert FinicToolkit(client=client, operations=()).get_tools() == []
        with pytest.raises(ValidationError):
            FinicToolkit(client=client, operations=("calibrate", "calibrate"))
        with pytest.raises(ValidationError):
            FinicToolkit(client=client, operations=("delete_score",))
        with pytest.raises(ValidationError):
            FinicScorer(model_id="scm_test", client=client, api_key="another")
    with scorer() as model:
        with pytest.raises(ValidationError):
            model.invoke(record() | {"model_id": "scm_other"})
        with pytest.raises(TypeError):
            model.invoke(record(), unknown=True)


def test_async_cancellation_closes_owned_pool():
    async def run():
        entered = asyncio.Event()

        class Transport(httpx.AsyncBaseTransport):
            closed = False

            async def handle_async_request(self, request):
                entered.set()
                await asyncio.Event().wait()

            async def aclose(self):
                self.closed = True

        transport = Transport()
        model = FinicScorer(model_id="scm_test", api_key="test", async_transport=transport)
        task = asyncio.create_task(model.ainvoke(record()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert transport.closed
        await model.aclose()

    asyncio.run(run())


def test_checkpointed_graph_resumes_wait_without_repeating_creation():
    from langgraph.checkpoint.memory import InMemorySaver

    build_graph = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/lifecycle_graph.py"))[
        "build_graph"
    ]

    async def run():
        calls = []
        failed_once = False

        def handle(request):
            nonlocal failed_once
            calls.append((request.method, request.url.path))
            if request.url.path == "/v1/scores":
                return httpx.Response(
                    202, json={"model_id": "scm_graph", "job_id": "job_create", "request_id": "req_create"}
                )
            if request.url.path == "/v1/jobs/job_create" and not failed_once:
                failed_once = True
                return httpx.Response(503)
            if request.url.path.startswith("/v1/jobs/"):
                return httpx.Response(
                    200, json={"job_id": request.url.path.rsplit("/", 1)[-1], "complete": True}
                )
            if request.url.path == "/v1/score":
                return score_response(request)
            if request.url.path == "/v1/calibrate":
                return httpx.Response(
                    202,
                    json={
                        "job_id": "job_calibrate",
                        "matched": 1,
                        "unmatched": 0,
                        "request_id": "req_calibrate",
                    },
                )
            raise AssertionError(request.url)

        async with FinicClient("test", max_attempts=1, async_transport=httpx.MockTransport(handle)) as client:
            graph = build_graph(client, InMemorySaver())
            config = {"configurable": {"thread_id": "synthetic"}}
            with pytest.raises(FinicError):
                await graph.ainvoke(
                    {
                        "examples": [{"amount": i, "is_bad": i % 2} for i in range(100)],
                        "record": record(),
                        "is_bad": False,
                    },
                    config,
                )
            checkpoint = await graph.aget_state(config)
            assert checkpoint.values["model_id"] == "scm_graph"
            assert checkpoint.values["creation_job_id"] == "job_create"
            assert checkpoint.next == ("wait_creation",)
            result = await graph.ainvoke(None, config)
        assert result["calibration_ready"] and result["creation_ready"]
        assert result["scoring_result"]["score_id"] == "record_0"
        assert calls.count(("POST", "/v1/scores")) == 1
        assert calls.count(("POST", "/v1/calibrate")) == 1

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_agent_example_tool_call(asynchronous):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    class Chat(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert {tool.name for tool in tools} == {
                "finic_score",
                "finic_list_scores",
                "finic_get_job_status",
            }
            return self

    chat = Chat(
        responses=[
            AIMessage(content="", tool_calls=[{"id": "score_call", "name": "finic_score", "args": record()}]),
            AIMessage(content="Review completed."),
        ]
    )
    build_agent = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/scoring_agent.py"))[
        "build_agent"
    ]
    transport = httpx.MockTransport(score_response)
    client = FinicClient("test", transport=transport, async_transport=transport)
    if asynchronous:

        async def run():
            async with client:
                return await build_agent(chat, client, "scm_test").ainvoke(
                    {"messages": [HumanMessage(content="Review this record.")]}
                )

        result = asyncio.run(run())
    else:
        with client:
            result = build_agent(chat, client, "scm_test").invoke(
                {"messages": [HumanMessage(content="Review this record.")]}
            )
    assert result["messages"][-2].tool_call_id == "score_call"
    assert json.loads(result["messages"][-2].content)["score_id"] == "record_0"


def test_scoring_cli_example(monkeypatch, capsys):
    original = FinicScorer.__init__

    def offline(self, **kwargs):
        original(self, api_key="test", transport=httpx.MockTransport(score_response), **kwargs)

    monkeypatch.setattr(FinicScorer, "__init__", offline)
    monkeypatch.setattr("sys.argv", ["scoring.py", "--model-id", "scm_test", "--score-id", "unique"])
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/scoring.py"), run_name="__main__")
    assert "unique" in capsys.readouterr().out
