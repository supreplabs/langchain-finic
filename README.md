# langchain-finic

**Add typed risk decisions to your LangChain agents and LangGraph workflows.**

[![PyPI version](https://img.shields.io/pypi/v/langchain-finic)](https://pypi.org/project/langchain-finic/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-finic)](https://pypistats.org/packages/langchain-finic)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-finic)](https://pypi.org/project/langchain-finic/)
[![CI](https://github.com/supreplabs/langchain-finic/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/supreplabs/langchain-finic/actions/workflows/ci.yml)
[![MIT license](https://img.shields.io/github/license/supreplabs/langchain-finic)](https://github.com/supreplabs/langchain-finic/blob/main/LICENSE)

[Documentation](https://supreplabs.github.io/langchain-finic/) ·
[Quickstart](#quickstart) · [Examples](#langchain-and-langgraph-examples) ·
[API reference](https://supreplabs.github.io/langchain-finic/api/) ·
[Releases](https://github.com/supreplabs/langchain-finic/releases)

Send transaction evidence to Finic Grayson, get structured answers, and route
cases through your own review policy. Use the same classifier in plain Python,
an LCEL chain, a LangGraph node, or a LangChain agent tool.

- **Triage a payment** with a typed probability or a choice among review actions.
- **Score records** with a custom Finic model and manage its calibration.
- **Add Kairos to an agent** for action decisions, event tracking, or named workflows.

Maintained by Suprep Labs. The current stable release is 0.1.0.
See the [compatibility and support policy](https://supreplabs.github.io/langchain-finic/compatibility/).

## Quickstart

Python 3.11 or newer and a Finic API key are required.

```bash
pip install langchain-finic==0.1.0
export FINIC_API_KEY=your_key
```

```python
from langchain_finic import DecisionRequest, FinicClassifier, Noul

request = DecisionRequest(
    data={"events": ["New device", "Payment to an unfamiliar recipient"]},
    questions={"takeover": Noul(instructions="Does the evidence suggest account takeover?")},
)
with FinicClassifier() as classifier:
    response = classifier.invoke(request)
    print(response.nouls["takeover"].noul)
```

The example sends fictional evidence to Finic and spends API credits. The result
is a model probability, not a verified fraud determination; your application
decides how to use it. The default model is `grayson-1`; set `FINIC_MODEL` to
change it. Applications are responsible for loading environment variables,
including values from `.env` files. See
[configuration and data handling](https://supreplabs.github.io/langchain-finic/usage/)
for client ownership, retries, and tracing.

## Capabilities

| Capability | Interface |
| --- | --- |
| Typed decisions | `Noul`, `Choice`, `MultiChoice`, `Score` |
| LCEL composition | `FinicClassifier` as a LangChain Runnable |
| Sync and async execution | `invoke`, `ainvoke`, `batch`, `abatch` |
| Agent tools | `classifier.as_tool()` with `create_agent` or `ToolNode` |
| Custom scoring | `FinicScorer`, `FinicToolkit`, and `FinicClient` |
| Kairos decisions and events | Optional `KairosRunnable` and `KairosToolkit` (`[kairos]` extra) |
| Optional middleware | `ModelRouterMiddleware` and `AutoModeMiddleware` |
| Host-enabled observability | LangSmith traces and usage metadata |

Batching makes individual API requests. Streaming yields one complete decision.
Use a host chat model for chat generation; tool screening remains probabilistic
and does not replace application authorization.

## LangChain and LangGraph examples

- [LCEL and LangGraph](https://github.com/supreplabs/langchain-finic/blob/main/examples/composition.py): compose a classifier and a typed graph node.
- [Finic decisions with LangChain](https://supreplabs.github.io/langchain-finic/tutorials/langchain-decisions/): use a decision in an LCEL chain.
- [LangGraph fraud review](https://supreplabs.github.io/langchain-finic/tutorials/langgraph-review/): route a fictional case through a graph.
- [Finic and a chat model with LangGraph](https://supreplabs.github.io/langchain-finic/tutorials/finic-langgraph-host-model/): route fictional payment evidence and draft an analyst note.
- [Agent middleware](https://github.com/supreplabs/langchain-finic/blob/main/examples/middleware_agent.py): route between host chat models and screen tool calls.
- [Custom scoring](https://supreplabs.github.io/langchain-finic/scoring/): use a scorer as a Runnable or tool, and manage score model lifecycle operations.
- [System 1 evaluation](https://supreplabs.github.io/langchain-finic/evaluation/): reproduce a small synthetic evaluation of routing and tool screening, including its limitations.
- [Usage guide](https://supreplabs.github.io/langchain-finic/usage/): async calls, batching, agent tools, and configuration.

For middleware, install the extra and the provider packages for your host chat models:

```bash
pip install 'langchain-finic[middleware]==0.1.0'
```

For Kairos event tracking and action decisions, install the optional extra:

```bash
pip install 'langchain-finic[kairos]==0.1.0'
```

See the [Kairos adapter guide](https://supreplabs.github.io/langchain-finic/kairos/).

## Frequently asked questions

### Is Finic Grayson a chat model in this package?

No. The package exposes Finic's structured decision and scoring APIs, not a
LangChain chat model. Use a separate chat model when your application needs
generated text or an agent that reasons and calls tools.

### Does batching send a single request to Finic?

No. LangChain batch methods issue individual Finic API requests. Account for
request limits, latency, and API credits when choosing concurrency.

### Can I use this package for fraud detection or AML review?

It can integrate Grayson decisions and custom scores into fraud or AML review
workflows. The package does not provide a validated fraud or AML detection
system, and probabilistic output is not a substitute for authorization or
human review where required.

## Contributing

If this integration saves you time, [star it on GitHub](https://github.com/supreplabs/langchain-finic).
If you try it in an application, tell us what worked and where you got stuck in
[the release feedback issue](https://github.com/supreplabs/langchain-finic/issues/12).


See [CONTRIBUTING.md](https://github.com/supreplabs/langchain-finic/blob/main/CONTRIBUTING.md) for setup, tests, and documentation standards.
Default tests run offline with mock providers. Report bugs through
[GitHub issues](https://github.com/supreplabs/langchain-finic/issues) and vulnerabilities
through [private security reporting](https://github.com/supreplabs/langchain-finic/blob/main/SECURITY.md).

Released under the [MIT license](https://github.com/supreplabs/langchain-finic/blob/main/LICENSE). See the [changelog](https://github.com/supreplabs/langchain-finic/blob/main/CHANGELOG.md) for version history.

Obtain a Finic API key through your [Finic account](https://finic.ai/) before
running live examples. Check your account access and credits; host chat models
use separate provider credentials.
