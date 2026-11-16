"""Execute published Python snippets using mock HTTP and deterministic chat models."""

import json
import re
import runpy
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from langchain_finic import DecisionRequest, FinicClassifier, Noul

ROOT = Path(__file__).resolve().parents[2]


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def chat(tool_name, args):
    return ToolCallingModel(
        responses=[
            AIMessage(content="", tool_calls=[{"id": "doc_call", "name": tool_name, "args": args}]),
            AIMessage(content="Reviewed fictional case."),
        ]
    )


@pytest.fixture
def offline_docs(monkeypatch):
    monkeypatch.setenv("FINIC_API_KEY", "documentation-test-key")
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        answers = {}
        for name, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.1}
            else:
                keys = list(question["criteria"])
                selected = keys[0]
                answers[name] = {
                    "type": "choice",
                    "choice": selected,
                    "confidence": 0.8,
                    "probabilities": {key: 0.9 if key == selected else 0.1 for key in keys},
                }
        return httpx.Response(
            200,
            json={
                "id": "dec_docs",
                "model": payload["model"],
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "request_id": "req_docs",
                "evaluation_time_ms": 1,
            },
        )

    original_init = FinicClassifier.__init__

    def offline_init(self, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(respond))
        kwargs.setdefault("async_transport", httpx.MockTransport(respond))
        original_init(self, **kwargs)

    monkeypatch.setattr(FinicClassifier, "__init__", offline_init)
    return calls


@pytest.mark.parametrize("page", ["README.md", "docs/index.md", "docs/usage.md"])
def test_documented_python_examples(page, offline_docs):
    namespace = {"__name__": "documentation_example"}
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / page).read_text(), re.DOTALL)
    assert blocks, f"No Python examples found in {page}"
    for index, block in enumerate(blocks):
        exec(compile(block, f"{page}:block-{index}", "exec"), namespace)
    assert offline_docs
    if "review_with_agent" in namespace:
        model = chat("finic_decide", namespace["request"].model_dump(mode="json"))
        result = namespace["review_with_agent"](model)
        assert result["messages"][-1].content == "Reviewed fictional case."
        assert result["messages"][-2].tool_call_id == "doc_call"
    if "review_with_middleware" in namespace:
        result = namespace["review_with_middleware"](
            chat("lookup_demo_case", {"case_id": "demo"}),
            chat("lookup_demo_case", {"case_id": "demo"}),
        )
        assert result["finic_model_route"]["selected"] == "fast"
        assert result["messages"][-2].tool_call_id == "doc_call"
        assert any("tool_risk" in call["questions"] for call in offline_docs)


def test_documented_graph_example(offline_docs):
    runpy.run_path(str(ROOT / "examples/composition.py"), run_name="__main__")
    assert len(offline_docs) == 2


@pytest.mark.parametrize("request_model", [None, "fictional-request-model"])
def test_documented_native_tool_arguments_keep_tags_and_model_precedence(request_model, offline_docs):
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/usage.md").read_text(), re.DOTALL)
    block = next(block for block in blocks if "def invoke_decision_tool(" in block)
    namespace = {}
    exec(compile(block, "docs/usage.md:native-tool-arguments", "exec"), namespace)
    fields = {} if request_model is None else {"model": request_model}
    request = DecisionRequest(
        data={"ticket": "Fictional support question"},
        questions={"urgent": Noul(instructions="Urgent review?")},
        **fields,
    )
    with FinicClassifier(model="fictional-configured-model") as classifier:
        result = namespace["invoke_decision_tool"](classifier, request)
    assert result["answers"]["urgent"]["noul"] == 0.1
    assert offline_docs[-1]["questions"]["urgent"]["type"] == "noul"
    assert offline_docs[-1]["model"] == (request_model or "fictional-configured-model")


def test_documented_offline_application_recipe():
    namespace = {"__name__": "documentation_example"}
    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/testing.md").read_text(), re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, "docs/testing.md:offline-recipe", "exec"), namespace)
    assert len(namespace["fixture_calls"]) == 1
    assert namespace["result"].nouls["urgent"].noul == 0.1


def test_relative_documentation_links_resolve():
    pages = [*ROOT.glob("*.md"), *ROOT.glob("docs/*.md")]
    for page in pages:
        for href in re.findall(r"\]\(([^)]+)\)", page.read_text()):
            if "://" in href or href.startswith("#"):
                continue
            assert (page.parent / href.split("#")[0]).exists(), (page.name, href)


@pytest.mark.asyncio
async def test_documented_score_reconciliation_does_not_replay_failed_writes():
    from langchain_finic import FinicScorer

    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body["score_id"])
        if body["score_id"] == "failed":
            return httpx.Response(409)
        return httpx.Response(
            200,
            json={
                "model_id": body["model_id"],
                "score_id": body["score_id"],
                "score": 200,
                "request_id": "req_test",
            },
        )

    blocks = re.findall(r"```python\n(.*?)\n```", (ROOT / "docs/scoring.md").read_text(), re.DOTALL)
    namespace = {}
    for block in blocks:
        if "async def score_records(" in block or "async def reconcile_scores(" in block:
            exec(compile(block, "docs/scoring.md:reconciliation", "exec"), namespace)
    namespace["FinicScorer"] = lambda **kwargs: FinicScorer(
        **kwargs, api_key="test", async_transport=httpx.MockTransport(handle)
    )
    results = await namespace["reconcile_scores"](
        "scm_test",
        [
            {"score_id": "failed", "data": {"amount": 1}},
            {"score_id": "ok", "data": {"amount": 2}},
        ],
    )
    assert results == [
        {"score_id": "failed", "status": "needs_reconciliation", "error_type": "FinicError"},
        {"score_id": "ok", "status": "scored", "score": 200},
    ]
    assert sorted(calls) == ["failed", "ok"]
