# Usage and configuration

The examples below use fictional evidence and require `FINIC_API_KEY` in the
process environment. Run Python snippets in order within each section. Calls to
Finic spend API credits; the package does not load `.env` files.

## Decisions and LCEL

```python
from langchain_finic import Choice, DecisionRequest, FinicClassifier, Noul

request = DecisionRequest(
    data={"events": ["New device", "Payment to an unfamiliar recipient"]},
    questions={
        "takeover": Noul(instructions="Does the evidence suggest account takeover?"),
        "route": Choice(
            instructions="Choose a review queue.",
            criteria={"standard": "Routine review", "urgent": "Urgent investigation"},
        ),
    },
)
with FinicClassifier(timeout=75, max_attempts=2) as classifier:
    decision = classifier.invoke(request)
    print(decision.nouls["takeover"].noul)
    route_chain = classifier | (lambda result: result.choices["route"].choice)
    print(route_chain.invoke(request, config={"tags": ["review"]}))
```

Each invocation accepts a `DecisionRequest` or a dictionary with `data` and/or `context`,
`questions`, and an optional `model`. State can be text, a JSON object or array,
or LangChain messages nested within those inputs. Messages are normalized to
role/content/tool-call JSON. Request data is sent to Finic.

| Question | Purpose | Response accessor |
| --- | --- | --- |
| `Noul` | Probability that a proposition is true | `response.nouls[name].noul` |
| `Choice` | Select one named option | `response.choices[name].choice` |
| `MultiChoice` | Select options whose probabilities reach 0.5 | `response.multi_choices[name].multi_choice` |
| `Score` | Expected index across ordered levels, starting at zero | `response.scores[name].score` |

`Choice` and `Score` answers include a probability distribution and confidence.
`Noul` returns the probability itself; `MultiChoice` returns a probability per
option. These model outputs do not guarantee accuracy for a particular dataset.
See the [API reference](api.md) for request fields and response types.

## Async, batching, and streaming

This example uses `request` from the preceding section. In a Python script, run
the coroutine with `asyncio.run`; in a notebook with an active event loop, use
`await review()` instead of `asyncio.run(review())`.

```python
import asyncio

async def review():
    async with FinicClassifier() as classifier:
        results = await classifier.abatch([request], config={"max_concurrency": 1})
        return [result.choices["route"].choice for result in results]

print(asyncio.run(review()))
```

`batch` and `abatch` issue a separate decision request for each input. They do not
use a provider-side bulk endpoint. `max_concurrency` limits concurrent tasks, not
requests per second; the service's organization quota still applies across callers.
A failed request raises by default; use `return_exceptions=True` when the caller
needs to handle individual failures. `stream` and `astream` yield one complete
response, not text tokens.

## Configuration

| Constructor option | Environment variable | Default / constraint |
| --- | --- | --- |
| `api_key` | `FINIC_API_KEY` | Required unless supplied through an injected client |
| `model` | `FINIC_MODEL` | `grayson-1`; nonempty, no whitespace |
| `timeout` | `FINIC_TIMEOUT_SECONDS` | 60 seconds; greater than 30 and at most 180 |
| `max_attempts` | `FINIC_MAX_ATTEMPTS` | 3 total attempts, including the first; allowed range 1–5 |

Explicit constructor values override environment defaults. Model precedence is:
invocation or `.bind(model=...)` override, explicitly supplied request model,
classifier constructor model, then environment/default. An omitted Pydantic request
model does not override the classifier's model. The service determines which model
names are supported. `GRAYSON_*` environment aliases are not supported.

Unknown constructor fields and unsupported invocation options raise errors.
`model` is the supported invocation override. LangChain configuration such as tags,
metadata, callbacks, and `max_concurrency` is passed through `config`.
`.configurable_fields(...)` supports standard LangChain configuration overrides.

### Client ownership

Use `with FinicClassifier()` or `async with FinicClassifier()` to reuse owned HTTP
pools across calls and batches. Outside a context, owned pools are closed after
each invocation, including on failure or cancellation. Use one event loop for an
async client lifetime. Close only after in-flight operations finish; call
`await aclose()` after async use, or `close()` for sync-only use.

Injected `FinicClient`, `httpx.Client`, and `httpx.AsyncClient` instances remain
caller-owned. A `client=` cannot be combined with other transport or credential
options. `http_client` conflicts with `transport`; `http_async_client` conflicts
with `async_transport`. The integration still applies authentication, timeout, and
redirect prevention to requests made with borrowed HTTP clients.

LangChain creates temporary models for configurable-field overrides. These use
invocation-scoped pools even if the original classifier is inside a context.
For frequent overrides, inject a shared caller-owned HTTP client. Owned pools
also close their transports; put reusable custom transports in a shared HTTP
client rather than passing the transport directly to multiple invocations.

### Errors and retries

`FinicError` represents provider failures, exhausted transport retries, or a
response that violates the decision contract. Its optional `status` and
`request_id` fields help diagnose service failures without printing case data.
Local request validation raises Pydantic `ValidationError`; unsupported invocation
options raise `TypeError`.

The client retries transport errors and HTTP 429, 500, 502, 503, and 504 with bounded
backoff. `timeout` is an HTTPX timeout per attempt, not a total invocation deadline.
Retrying a timed-out successful decision can incur another charge. Avoid stacking
independent retry policies without considering the combined attempt count.

Using the request from the first section:

```python
from langchain_finic import FinicError

try:
    with FinicClassifier(max_attempts=1) as classifier:
        decision = classifier.invoke(request)
except FinicError as error:
    print("Decision unavailable", error.status, error.request_id)
else:
    print(decision.choices["route"].choice)
```

## LangGraph

Install `langgraph>=1,<2` in the host application. The complete
[composition example](https://github.com/supreplabs/langchain-finic/blob/main/examples/composition.py)
shows a typed graph state, a decision node, LCEL composition, and client cleanup.

The tool returned by `.as_tool()` also works in LangGraph `ToolNode` and in
`create_agent`. It converts the decision to JSON-compatible output and preserves
the tool-call ID when returning a `ToolMessage`.

## Tool-calling agents

A host chat model decides when to call the Finic tool. Install its provider package
and configure its credentials in addition to `FINIC_API_KEY`. Grayson is the
decision tool; it does not implement chat generation or `bind_tools`.

The following reusable function requires an initialized `BaseChatModel` that
supports tool calling. It defines all other inputs locally:

```python
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_finic import FinicClassifier


def review_with_agent(chat_model: BaseChatModel):
    with FinicClassifier() as classifier:
        agent = create_agent(model=chat_model, tools=[classifier.as_tool()])
        return agent.invoke({"messages": [{
            "role": "user",
            "content": "Use finic_decide to assess this fictional case: a new device sent a payment to a new payee.",
        }]})
```

## Agent middleware

```bash
pip install 'langchain-finic[middleware]==0.1.0a6'
# Also install the provider package for your host chat models.
```

This function accepts two initialized host chat models with tool-calling support.
The only tool reads a fictional local record; it performs no external action.

```python
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_finic import FinicClassifier
from langchain_finic.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware


@tool
def lookup_demo_case(case_id: str) -> dict:
    """Look up a fictional local case by ID."""
    return {"case_id": case_id, "status": "awaiting independent review"}


def review_with_middleware(fast_model: BaseChatModel, powerful_model: BaseChatModel):
    with FinicClassifier() as classifier:
        router = ModelRouterMiddleware(
            classifier=classifier,
            choices={
                "fast": ModelChoice(model=fast_model, criteria="Simple lookups and extraction."),
                "powerful": ModelChoice(model=powerful_model, criteria="Complex evidence analysis."),
            },
            minimum_probability=0.70,
            fallback="powerful",
        )
        guard = AutoModeMiddleware(classifier=classifier, tools=[lookup_demo_case], threshold=0.5)
        agent = create_agent(fast_model, tools=[lookup_demo_case], middleware=[router, guard])
        return agent.invoke({"messages": [{"role": "user", "content": "Look up the demo case."}]})
```

`ModelChoice.model` also accepts a provider-qualified model string resolved by
LangChain's `init_chat_model`. `FINIC_MODEL` selects the decision model, not either
host chat model. The [command-line example](https://github.com/supreplabs/langchain-finic/blob/main/examples/middleware_agent.py)
accepts those provider-qualified strings as arguments.

`ModelRouterMiddleware` classifies the latest human message in `before_agent`
and uses the selected model throughout that invocation's tool loop. It records
the selection and answer in `finic_model_route`. The next user invocation is
classified again, including with a checkpointer. Concurrent runs store their
routes in graph state rather than on the shared middleware instance.

If the winning option's probability is below `minimum_probability`, or a provider
or validation error occurs during classification, the configured `fallback` is
used. Without a fallback, the run stops. An invocation with no human message
raises an error. Unconfigured model names are never selected.

`AutoModeMiddleware` screens only the listed tools in `wrap_tool_call`. It sends
conversation messages and the proposed tool name/arguments to Finic. At or above
`threshold`, execution is skipped and an error `ToolMessage` retains the original
call ID. Classifier exceptions and malformed responses also block execution.
Unlisted tools run without a Finic assessment. Actual tool exceptions and
cancellation propagate to the host. Guarding a tool does not register it with the
agent; include it in the agent's `tools` list too.

Both middleware support native async hooks and context managers. An injected
classifier remains caller-owned. A middleware-created classifier closes when the
middleware context exits; keep that context alive for the agent's lifetime.
The guard is probabilistic screening, not deterministic authorization or human
approval. Retain application access checks and use
[human-in-the-loop middleware](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
when an action needs approval.

## Tracing and serialization

When the host enables LangSmith tracing, the classifier records an LLM run with
`ls_provider=finic`, the requested model name, token usage, request ID, and server
evaluation time. Callbacks, tags, and metadata propagate through LCEL and tools.
The library does not enable tracing automatically or calculate monetary cost.

The configured API key is excluded from representations and Pydantic dumps;
LangChain serialization stores a `FINIC_API_KEY` secret reference rather than its
value. Request state and caller metadata are not automatically redacted and may
appear in traces. Choose the data you send and your tracing retention accordingly.

Load serialized configurations only from trusted sources, supplying the secret
and the loader's class allowlist where required. Configurations containing custom
clients or transports are not serializable. See [security reporting and data handling](https://github.com/supreplabs/langchain-finic/blob/main/SECURITY.md).

## Migrating decision inputs

The current Finic API accepts `data` (JSON object, array, or nonblank text),
`context` (nonblank text), or both. LangChain messages nested in `data` are
normalized to JSON. Existing `state=` inputs and the `.state` accessor remain
supported as compatibility aliases for `data`. Serialization and tool schemas
use `data`; passing both `state` and `data` raises a validation error.
Legacy `request.state = ...` and `request.model_copy(update={"state": ...})`
update the canonical evidence after validation. Invalid legacy updates leave the
original request unchanged. Default decision tools, explicit `DecisionRequest`
tool schemas, and tools built after `bind()`/`with_config()` retain this input
compatibility on the minimum supported LangChain version.

Noul and MultiChoice base rates must be strictly between zero and one. Choice
and Score rates may sum to one within 0.02; Choice rates are normalized to one.
`FinicError.error_type` and `.param` expose bounded provider identifiers alongside
`.status` and `.request_id`. The default error message omits provider messages
and bodies; treat these identifiers and request traces as potentially sensitive.
