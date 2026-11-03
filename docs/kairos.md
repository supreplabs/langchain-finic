---
description: Use Finic Kairos event tracking and action decisions with LangChain and LangGraph.
---

# Kairos with LangChain

The optional Kairos adapter makes the `finic-kairos` SDK composable with LangChain.
It is separate from `FinicClassifier`: `KairosRunnable` calls a defined Kairos
action, while `KairosToolkit` can expose explicitly selected event and workflow
operations to a LangChain agent.

## Install

```bash
pip install 'langchain-finic[kairos]'
export KAIROS_API_KEY=your_secret_key
```

The adapter does not load `.env` files. Create and own the Kairos client in your
application, and keep it open while its Runnable or tools are in use:

```python
import os

from kairos import Kairos
from langchain_finic import KairosDecisionInput, KairosRunnable

client = Kairos(api_key=os.environ["KAIROS_API_KEY"])
decide = KairosRunnable(client=client)

result = decide.invoke(
    KairosDecisionInput(
        action="bank_transfer",
        properties={"user_id": "user_123", "amount": 9500, "currency": "USD"},
        fallback="review",
    )
)
print(result.decision, result.risk_score, result.reasons)

# On application shutdown:
client.shutdown()
```

This makes a live Kairos request and may consume service usage. `fallback` is an
application policy: choose it explicitly when Kairos should return an outcome if
it cannot answer before the timeout. Without it, SDK errors propagate.

`KairosRunnable` supports `invoke`, `ainvoke`, and standard Runnable composition.
The underlying Kairos Python SDK is synchronous, so `ainvoke` runs its call in a
worker thread. Use `as_tool()` when you want the typed decision to be one agent
tool; its output is JSON-safe. The Runnable borrows the client and never closes or
serializes it.

## Agent tools

`KairosToolkit` defaults to the `decide` tool. Add event tracking or workflow runs
only when the agent is meant to write that information or start those operations:

```python
import os

from kairos import Kairos
from langchain.agents import create_agent
from langchain_finic import KairosToolkit

client = Kairos(api_key=os.environ["KAIROS_API_KEY"])
tools = KairosToolkit(
    client=client,
    operations=("decide", "track", "track_and_decide", "flush"),
).get_tools()
agent = create_agent(model=host_chat_model, tools=tools)
```

Available operations are `decide`, `track`, `track_and_decide`, `run_workflow`,
and `flush`. `track` queues an event for the SDK's background sender and reports
that it was queued; it does not confirm server delivery. Select `flush` when the
application needs to wait for queued events to be sent. `track_and_decide` records
the triggering event and requests its action decision in one call. All async tool
methods offload the synchronous SDK call to a worker thread.

Tools expose Pydantic argument schemas, return JSON-safe results, and propagate SDK
errors. The toolkit borrows the client; call `shutdown()` on it when the application
stops. Do not let an agent select or execute a business outcome by itself: use the
decision as input to your application's authorization, review, or escalation policy.

## Composition

The Runnable can sit inside an LCEL chain or a LangGraph node:

```python
from langchain_core.runnables import RunnableLambda
from langchain_finic import KairosRunnable

decision_chain = request_builder | KairosRunnable(client=client) | RunnableLambda(
    lambda result: {"outcome": result.decision, "risk_score": result.risk_score}
)
```

Use the [API reference](api.md#kairos-runnable-and-toolkit) for schemas and the
[compatibility policy](compatibility.md) for the supported contract.
