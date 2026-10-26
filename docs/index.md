# langchain-finic

Community-maintained LangChain integration for **Finic Grayson**, by Suprep Labs.
Typed decisions, LCEL composition, LangGraph nodes, structured tools, and optional
agent middleware. This alpha is not endorsed by Finic or LangChain.

[Documentation](https://supreplabs.github.io/langchain-finic/) ·
[Source](https://github.com/supreplabs/langchain-finic) ·
[PyPI](https://pypi.org/project/langchain-finic/) ·
[MIT license](https://github.com/supreplabs/langchain-finic/blob/main/LICENSE)

## Install

Python 3.11 or newer. Install the alpha release:

```bash
pip install langchain-finic==0.1.0a4
export FINIC_API_KEY=your_key
# Optional; defaults to grayson-1:
export FINIC_MODEL=grayson-1
```

## Make a typed decision

```python
from langchain_finic import Choice, DecisionRequest, FinicClassifier, Noul

request = DecisionRequest(
    state={"events": ["New device", "Transfer to an unfamiliar recipient"]},
    questions={
        "takeover": Noul(instructions="Does this evidence suggest account takeover?"),
        "route": Choice(
            instructions="Choose a review queue.",
            criteria={"standard": "Routine review", "urgent": "Urgent investigation"},
        ),
    },
)

with FinicClassifier() as classifier:
    decision = classifier.invoke(request)
    print(decision.nouls["takeover"].noul)
    print(decision.choices["route"].choice)
    print(decision.request_id)
```

`Noul`, `Choice`, `MultiChoice`, and `Score` have typed answers. Responses retain
probabilities, usage, and request metadata. Finic evaluates decisions; it does not
generate chat messages or text tokens.

## Compose with LangChain and LangGraph

```python
# request is defined above.
with FinicClassifier() as classifier:
    route_chain = classifier | (lambda result: result.choices["route"].choice)
    route = route_chain.invoke(request, config={"tags": ["risk-review"]})
    tool = classifier.as_tool()
    print(tool.name)  # finic_decide; pass it to create_agent or ToolNode
```

```python
import asyncio

async def review():
    async with FinicClassifier() as classifier:
        return await classifier.abatch([request], config={"max_concurrency": 1})

results = asyncio.run(review())
```

Batching makes individual API requests; concurrency is controlled by Runnable config.
Streaming yields one complete decision. A chat-provider model can call the Finic tool
through `create_agent`; the Finic integration itself is a decision Runnable.

## Optional middleware

```bash
pip install 'langchain-finic[middleware]==0.1.0a4'
```

`ModelRouterMiddleware` selects among host chat models. `AutoModeMiddleware` screens
specified tools before execution. See the complete
[middleware example](https://github.com/supreplabs/langchain-finic/blob/main/examples/middleware_agent.py)
and [configuration guide](https://supreplabs.github.io/langchain-finic/usage/).
Tool screening is probabilistic; application authorization remains necessary.

## Configuration and data

Use `FINIC_API_KEY`, `FINIC_MODEL`, `FINIC_TIMEOUT_SECONDS`, and `FINIC_MAX_ATTEMPTS`,
or constructor options. The library does not load `.env`. The host controls optional
LangSmith tracing, which can record request data and responses. Default timeout is
60 seconds, with up to three attempts; retries can incur additional charges.
Use context managers to reuse and close owned HTTP clients. Injected clients remain
caller-owned. Read the [usage guide](https://supreplabs.github.io/langchain-finic/usage/)
for precedence, serialization, resource lifetimes, and error handling.

## Development

```bash
uv sync --locked --group dev --extra middleware
make check
```

Default tests use mocked HTTP with network disabled. Opt-in live tests spend Finic
credits. See [contributing](https://github.com/supreplabs/langchain-finic/blob/main/CONTRIBUTING.md)
and [security reporting](https://github.com/supreplabs/langchain-finic/blob/main/SECURITY.md).

References: [Finic API](https://docs.finic.ai/reference/decide),
[LangChain integration guidelines](https://docs.langchain.com/oss/python/contributing/integrations-langchain).
