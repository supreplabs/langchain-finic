# Usage and configuration

## Configuration and ownership

`model`, `api_key`, `timeout`, `max_attempts`, `client`, `http_client`,
`http_async_client`, `transport`, and `async_transport` are typed constructor fields.
Unknown fields fail validation. Timeout and attempts also read
`FINIC_TIMEOUT_SECONDS` and `FINIC_MAX_ATTEMPTS`. Explicit constructor values win.
`FINIC_MODEL` defaults to `grayson-1`. Model precedence is: invocation/bound override,
explicit request model, classifier constructor model, then `FINIC_MODEL`/default.
An omitted Pydantic request model does not override the classifier constructor.
Model names must be nonempty and contain no whitespace; the Finic service decides
which names are supported. The integration does not claim other models are available.

```python
with FinicClassifier(timeout=75, max_attempts=2) as classifier:
    response = classifier.bind(model="grayson-1").invoke(request)
    probability = response.nouls["takeover"].noul

async with FinicClassifier() as classifier:
    response = await classifier.ainvoke(request)
```

Inside `with` / `async with`, owned HTTP pools are created lazily and reused across
calls and batches. Outside a context, each invocation closes its owned pool on
success, failure, or cancellation. `close()`
closes sync resources; after async use call `await aclose()` on the same event loop.
Close only after in-flight operations finish. Injected HTTP clients and an injected
`FinicClient` remain caller-owned. Request authentication and redirect prevention
still apply to injected HTTP clients.

`.configurable_fields(...)` supports standard LangChain configurable fields.
LangChain creates temporary models for overrides; these use invocation-scoped
pools even when the original model is inside a context manager. For
frequent per-call reconfiguration, inject a shared, caller-owned HTTP client to
manage the resulting configurations' connection lifetime explicitly. Invocation
kwargs currently support `model`; unsupported options fail clearly.
Temporary owned pools close their transports too; wrap a reusable custom transport
in an injected HTTP client when sharing it across configurations.

The state accepts nested LangChain messages and converts them to role/content/tool
call dictionaries. `.nouls`, `.choices`, `.multi_choices`, `.scores`, and
`.usage_metadata` provide typed response accessors. `.as_tool(args_schema=None, *,
name=None, description=None, arg_types=None)` supports the standard Runnable tool
conversion signature, including JSON ToolMessage output.

## Tool-calling agents

Install your chat provider and `langchain>=1` in the host application. Grayson is
the decision tool; the host chat model chooses when to call it:

```python
from langchain.agents import create_agent

with FinicClassifier() as classifier:
    agent = create_agent(model=chat_model, tools=[classifier.as_tool()])
    result = agent.invoke({"messages": [{"role": "user", "content": "Review this payment evidence..."}]})
```

The tool also works in LangGraph `ToolNode`, with checkpointed message state.
Both synchronous and asynchronous `create_agent` execution are covered by mock
transport integration tests. Grayson does not implement chat generation,
`bind_tools`, or token streaming; use a chat provider for those capabilities.

## Tracing and serialization

Host-enabled LangSmith tracing receives an LLM run with `ls_provider=finic`,
the actual requested `ls_model_name`, token usage, request ID, and server evaluation time.
Caller callbacks, tags, run names, and metadata propagate through LCEL and tools.
No tracing service is enabled automatically. No dollar-cost estimate is invented.

LangChain serialization stores the key as a `FINIC_API_KEY` secret reference,
never its value. Load only trusted serialized configurations, supplying that secret
and allowing `FinicClassifier` in the loader's class allowlist where required.
Custom clients/transports are deliberately not serializable because reloading them
would change their behavior. Pydantic configuration dumps exclude keys and clients.

## Test coverage

The suite exercises request/response contracts, sync/async retries, configuration,
serialization, messages, batching, callbacks, tool calls, resource ownership,
cancellation, LangGraph checkpoints, and LangChain `create_agent`. Provider HTTP
is mocked; tests neither read `.env` nor spend credits. Chat-model standard tests
are not applicable to this decision Runnable. The package is an alpha maintained by Suprep Labs.


## Agent middleware

Install the optional dependencies and your chat-model provider integrations:

```bash
pip install 'langchain-finic[middleware]'
# Also install the provider packages needed by your configured chat models.
```

```python
from langchain.agents import create_agent
from langchain_finic import FinicClassifier
from langchain_finic.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware

# fast_model and powerful_model are initialized BaseChatModel instances.
# review_tool is a host-provided LangChain tool; it must also be in the agent's tools.
with FinicClassifier() as classifier:
    router = ModelRouterMiddleware(
        classifier=classifier,
        choices={
            "fast": ModelChoice(model=fast_model, criteria="Simple lookups and extraction."),
            "powerful": ModelChoice(model=powerful_model, criteria="Complex analysis and reasoning."),
        },
        minimum_probability=0.70,
        fallback="powerful",
    )
    guard = AutoModeMiddleware(
        classifier=classifier,
        tools=[review_tool],
        threshold=0.5,
        criteria={
            "true": "Risky, outside the user's scope, or insufficiently authorized.",
            "false": "Within the user's request and acceptable under the tool policy.",
        },
    )
    agent = create_agent(fast_model, tools=[review_tool], middleware=[router, guard])
    result = agent.invoke({"messages": [{"role": "user", "content": "Review this case."}]})
    print(result["finic_model_route"])
```

`ModelChoice.model` also accepts a provider-qualified model string, initialized
through LangChain's `init_chat_model`. Provider dependencies and credentials belong
to the host application. `FINIC_MODEL` chooses the Finic decision model, not either
host chat model.

`ModelRouterMiddleware` classifies the latest human message in `before_agent`,
then uses that selected model for every call in the invocation's tool loop. The
selected name and full typed answer are stored as JSON in `finic_model_route`.
The next user invocation is classified again, including with a checkpointer.
Concurrent runs hold their choices in separate graph state. No route is stored
on the shared middleware instance. If the winning option's probability is below
`minimum_probability`, or classification fails, an explicitly configured `fallback`
is used. With no fallback, the run stops. It never selects an unconfigured model.

`AutoModeMiddleware` assesses only the listed tool names/objects in
`wrap_tool_call`. It sends conversation messages and the exact tool name/arguments
to Finic. At or above `threshold`, execution is skipped and an error `ToolMessage`
with the original call ID is returned. Malformed responses and classifier errors
also block execution without exposing error details. Unlisted tools pass through
without a Finic call. Actual tool exceptions propagate to the host; cancellation
propagates normally. Guarding a tool does not register it with the agent.

Both middleware implement native async hooks, preserve callback/tracing context,
and support `with`/`async with`. A supplied classifier remains caller-owned; an
internally created classifier is closed when its middleware context exits. Keep
that context alive for the agent's lifetime.

The guard is probabilistic screening, not deterministic authorization or a human
approval system. Retain application access checks; use LangChain's human-review
middleware when an action requires approval. Conversation and tool arguments sent
to the guard must be appropriate to share with the decision provider.

The optional module is `langchain_finic.middleware`; importing `langchain_finic`
itself does not import LangChain agents or require the middleware extra. See
[the runnable host example](https://github.com/supreplabs/langchain-finic/blob/main/examples/middleware_agent.py). The middleware
uses LangChain's [documented hooks](https://docs.langchain.com/oss/python/langchain/middleware/custom).

## Batch and streaming behavior

`batch` and `abatch` issue one request per input; there is no server-side batch endpoint.
Set `max_concurrency` in Runnable config to respect your organization quota.
`stream` and `astream` yield a complete decision, not text tokens.

## LCEL and LangGraph

See the self-contained [composition example](https://github.com/supreplabs/langchain-finic/blob/main/examples/composition.py).
Install `langgraph>=1,<2` in your application to run it.
