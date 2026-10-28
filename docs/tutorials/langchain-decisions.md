---
title: Finic Grayson decisions with LangChain
description: Use Finic Grayson with a LangChain Runnable and LCEL to assess fictional account evidence.
---

# Use Finic Grayson decisions with LangChain

This tutorial sends fictional account evidence to Finic and uses a LangChain
Expression Language (LCEL) chain to extract the probability of account takeover.
`FinicClassifier` is a Runnable; Grayson supplies a decision, not generated chat.

## Before you begin

Use Python 3.11 or newer and a Finic API key. Calls send the evidence to Finic
and consume API credits; retries may create additional requests. The library does
not automatically load `.env` files.

```bash
pip install langchain-finic==0.1.0a7
export FINIC_API_KEY=your_key
```

## Build and invoke a chain

Save this code as `review.py` and run `python review.py`:

```python
from langchain_finic import DecisionRequest, FinicClassifier, Noul

request = DecisionRequest(
    data={"events": ["New device", "Payment to an unfamiliar recipient"]},
    questions={"takeover": Noul(instructions="Does the evidence suggest account takeover?")},
)

with FinicClassifier() as classifier:
    probability = classifier | (lambda result: result.nouls["takeover"].noul)
    print("Takeover probability:", probability.invoke(request))
```

This performs one decision invocation (which may retry). The result is a model
probability, not a confirmed fraud finding. Decide how to use it in your own
application, and evaluate it on representative data. Avoid sending unnecessary
customer information and consider the data captured by any enabled tracing.

Next: [configuration and tracing](../usage.md), [custom scoring](../scoring.md),
or the [LangGraph review tutorial](langgraph-review.md).
