---
title: Finic Grayson and a chat model in LangGraph for payment review
description: Build a Python LangGraph workflow that combines Finic Grayson decisions with a host chat model to draft a payment review note.
---

# Build a payment review workflow with Finic, LangGraph, and a chat model

This tutorial combines a Finic Grayson decision with a separate LangChain chat
model in a LangGraph workflow. It assesses fictional payment evidence, assigns a
review queue using an illustrative rule, and asks the chat model to draft a note
for an analyst. The [complete executable example](https://github.com/supreplabs/langchain-finic/blob/main/examples/finic_host_review.py)
contains the graph and its entry point.

The graph makes a Finic decision call, applies a local rule, then makes a host
chat-model call. It does not call tools, execute payments, investigate cases, or
confirm fraud. Both providers receive the fictional evidence; if tracing is
configured, traces may also contain that evidence and the generated note.

## Install and configure

Use Python 3.11 or later, a Finic API key, and credentials for a separate chat
model provider. The example below uses OpenAI as a host; another supported
LangChain provider can be used by installing its integration and setting its
credentials. Pick a model available to your provider account. Both providers may
charge for requests, and Finic's retry policy can send additional requests.

```bash
pip install 'langchain-finic==0.1.0a10' 'langchain>=1,<2' 'langgraph>=1,<2' langchain-openai
export FINIC_API_KEY=your_finic_key
export OPENAI_API_KEY=your_host_provider_key
```

The package reads environment variables; it does not load `.env` automatically.
`FINIC_MODEL` selects the Finic decision model (default `grayson-1`); the command
argument selects the separate host chat model.

## Run the workflow

Clone the [repository](https://github.com/supreplabs/langchain-finic), then run:

```bash
git clone https://github.com/supreplabs/langchain-finic.git
cd langchain-finic
python examples/finic_host_review.py openai:YOUR_AVAILABLE_MODEL
```

Replace `YOUR_AVAILABLE_MODEL` with a chat model available to your account. The
printed JSON includes the fictional evidence, the Finic probability and request
ID, the chosen queue, and the generated note. Model output varies between runs.

The graph's nodes have separate responsibilities:

1. **Assess:** `FinicClassifier` sends the evidence with a typed `Noul` question
   and returns a model probability.
2. **Choose a queue:** Python compares that probability with `0.7` and chooses
   `urgent` or `standard`. This number is illustrative, not a validated threshold.
3. **Draft a note:** the host chat model receives the evidence, probability, and
   queue, and drafts a note. The prompt asks it to treat the case as data and
   avoid claiming that fraud is confirmed.

## Follow the decision through the graph

The example starts with fictional customer context: a 760-day-old account, a new
device, and a payment to an unfamiliar recipient. These are the only case facts
provided to the models. A real review might require more context, such as the
customer's usual devices, prior recipients, and transaction history; include
only information appropriate for the providers and the review purpose.

The `assess` node builds a `DecisionRequest` and asks Grayson a typed `Noul`
question. The response probability and request ID become graph state:

```python
decision = classifier.invoke(
    DecisionRequest(
        data=state["evidence"],
        questions={"takeover": Noul(instructions="Does the evidence suggest account takeover?")},
    ),
    config=config,
)
return {"probability": decision.nouls["takeover"].noul, "request_id": decision.request_id}
```

The next node compares the probability with an **illustrative** threshold and
stores a queue. For example, a probability of `0.8` would select `urgent`; a
probability of `0.4` would select `standard`. These are examples of the rule,
not expected model outputs or validated fraud probabilities.

```python
queue = "urgent" if state["probability"] >= 0.7 else "standard"
return {"queue": queue}
```

Finally, `draft_note` sends the evidence, probability, and queue to the separate
chat model. The graph connects the three nodes in order:

```python
builder.add_edge(START, "assess")
builder.add_edge("assess", "choose_queue")
builder.add_edge("choose_queue", "draft_note")
builder.add_edge("draft_note", END)
```

The request ID can help correlate the Finic response with a review record; follow
your organization's data retention and access policies when storing it. The
queue is a recommendation for analyst attention, not a payment block or a fraud
finding.

The system prompt is not a security boundary. Host-model output can be inaccurate
or manipulated by case content. Applications should validate outputs, apply
independent authorization and human review, and send only data appropriate for
both providers. Decide how to handle failures and retention before using real data.
The example propagates provider errors; if Finic fails, the graph stops before
calling the chat model. A chat-model failure stops the graph rather than producing
a note. It has no durable checkpoint, so rerunning after a failure repeats earlier
calls and can incur additional charges.

## Adapt and evaluate

Replace the example rule only after evaluating decisions on representative,
properly labeled data and choosing an operating point for your review capacity.
For provider retries, configuration, and tracing, see [usage](../usage.md); for
limitations and evaluation design, see [System 1 evaluation](../evaluation.md).
The repository's offline tests run the graph against a mocked Finic response and
fake chat model; they verify control flow, not fraud detection accuracy.
