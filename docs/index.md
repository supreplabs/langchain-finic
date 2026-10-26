# Finic for LangChain

`langchain-finic` connects Finic Grayson's typed decisions to LangChain and
LangGraph. It is an MIT-licensed, community-maintained Python integration by
Suprep Labs. The package is alpha and is not endorsed by Finic or LangChain.

## Install

Use Python 3.11 or later and a Finic API key:

```bash
pip install langchain-finic==0.1.0a4
export FINIC_API_KEY=your_key
```

The default decision model is `grayson-1`. Set `FINIC_MODEL` or pass `model=` to
configure it. The application is responsible for loading environment variables.

## First decision

This example sends fictional evidence to Finic and spends API credits.

```python
from langchain_finic import DecisionRequest, FinicClassifier, Noul

request = DecisionRequest(
    state={"events": ["New device", "Payment to an unfamiliar recipient"]},
    questions={"takeover": Noul(instructions="Does the evidence suggest account takeover?")},
)
with FinicClassifier() as classifier:
    response = classifier.invoke(request)
    print(response.nouls["takeover"].noul)
    print(response.request_id)
```

The response contains model probabilities, not a verified fraud determination.
Applications choose their own review or routing policy.

## Supported interfaces

| Capability | Interface |
| --- | --- |
| Typed decisions | `Noul`, `Choice`, `MultiChoice`, `Score` |
| LCEL composition | `FinicClassifier` is a LangChain Runnable |
| Sync and async calls | `invoke`, `ainvoke`, `batch`, `abatch` |
| Agent tools | `FinicClassifier.as_tool()` |
| LangGraph | Runnable nodes or tools in `ToolNode` |
| Optional middleware | Model routing and tool screening |
| Host-enabled tracing | LangSmith metadata and usage |

Batching makes individual API requests. Streaming yields one complete response.
Use a separate chat-provider model for chat generation and agent orchestration.

Continue with the [usage guide](usage.md) for complete composition examples,
configuration, retries, client ownership, and data handling. The [API reference](api.md)
describes the public types and methods.

## Development and support

Report reproducible bugs through [GitHub issues](https://github.com/supreplabs/langchain-finic/issues).
See [contributing](https://github.com/supreplabs/langchain-finic/blob/main/CONTRIBUTING.md)
for development and test commands, and [security reporting](https://github.com/supreplabs/langchain-finic/blob/main/SECURITY.md)
for private vulnerability reports.

Provider reference: [Finic decision API](https://docs.finic.ai/reference/decide).
