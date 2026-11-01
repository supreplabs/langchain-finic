---
description: Get started using Finic Grayson decision and scoring APIs with Python, LangChain, and LangGraph.
---

# Finic Grayson for LangChain and LangGraph

`langchain-finic` connects Finic Grayson's typed decisions to LangChain and
LangGraph. It is an independent, MIT-licensed Python integration maintained by
Suprep Labs. The package is alpha and is not endorsed by Finic or LangChain.

[PyPI package](https://pypi.org/project/langchain-finic/) ·
[Source code](https://github.com/supreplabs/langchain-finic) ·
[Release history](https://github.com/supreplabs/langchain-finic/releases)

## Install

Use Python 3.11 or later and a Finic API key:

```bash
pip install langchain-finic==0.1.0a8
export FINIC_API_KEY=your_key
```

The default decision model is `grayson-1`. Set `FINIC_MODEL` or pass `model=` to
configure it. The application is responsible for loading environment variables.

## First decision

This example sends fictional evidence to Finic and spends API credits.

```python
from langchain_finic import DecisionRequest, FinicClassifier, Noul

request = DecisionRequest(
    data={"events": ["New device", "Payment to an unfamiliar recipient"]},
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

Try the [LangChain decision tutorial](tutorials/langchain-decisions.md) or the
[LangGraph fraud-review tutorial](tutorials/langgraph-review.md) for complete
examples using fictional evidence. To combine a decision with a host chat model,
try the [LangGraph payment-review workflow](tutorials/finic-langgraph-host-model.md).

## Frequently asked questions

### How do I use Finic Grayson in a LangChain or LangGraph application?

Install `langchain-finic`, set `FINIC_API_KEY`, and pass a `DecisionRequest` to
`FinicClassifier`. The classifier is a LangChain Runnable: compose it with LCEL,
use it in a LangGraph node, or turn it into an agent tool with `as_tool()`.
See the [LangChain tutorial](tutorials/langchain-decisions.md) and
[LangGraph tutorial](tutorials/langgraph-review.md).

### Is Grayson a LangChain chat model?

This integration exposes Finic's structured decisions and custom scoring, not a
LangChain chat model. Use a separate chat provider when an application needs
generated text or agent orchestration. The optional middleware can route between
host chat models and screen tool calls; see the [middleware guide](usage.md#agent-middleware).

### Does batching combine cases into one Finic API request?

No. `batch` and `abatch` make separate requests for individual inputs.
`max_concurrency` limits concurrent tasks rather than requests per second, and
retries may result in additional requests. See [batching and streaming](usage.md#async-batching-and-streaming).

### Can I use Finic Grayson for fraud detection or AML review?

The package can integrate decisions and scores into fraud or AML review workflows;
it is not itself a validated detection system. Evaluate a workflow on representative
data and define review and authorization policies for its intended use. See the
[evaluation guide](evaluation.md) and [custom scoring guide](scoring.md).

## Development and support

Report reproducible bugs through [GitHub issues](https://github.com/supreplabs/langchain-finic/issues).
See [contributing](https://github.com/supreplabs/langchain-finic/blob/main/CONTRIBUTING.md)
for development and test commands, and [security reporting](https://github.com/supreplabs/langchain-finic/blob/main/SECURITY.md)
for private vulnerability reports.

Provider reference: [Finic decision API](https://docs.finic.ai/reference/decide).
