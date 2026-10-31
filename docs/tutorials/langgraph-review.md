---
title: Build a Finic fraud-review workflow with LangGraph
description: Use a Finic Grayson decision in a LangGraph node to route fictional payment evidence for review.
---

# Build a fraud-review workflow with Finic and LangGraph

This example places a Finic Grayson decision in a LangGraph node and routes a
fictional case to a review queue. The graph provides orchestration; the model's
choice is probabilistic and does not establish whether fraud occurred.

## Before you begin

Install Python 3.11 or newer, the integration, and LangGraph. Set a Finic API key
in the process environment; the package does not load `.env` automatically.
Running the graph sends the fictional evidence to Finic and consumes API credits.
Retries may create additional requests.

```bash
pip install langchain-finic==0.1.0a8 'langgraph>=1,<2'
export FINIC_API_KEY=your_key
```

## Create the review graph

Save as `graph_review.py` and run `python graph_review.py`:

```python
from typing import NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from langchain_finic import Choice, DecisionRequest, FinicClassifier


class ReviewState(TypedDict):
    events: list[str]
    queue: NotRequired[str]


with FinicClassifier() as classifier:
    def assess(state: ReviewState, config: RunnableConfig):
        request = DecisionRequest(
            data={"events": state["events"]},
            questions={
                "queue": Choice(
                    instructions="Choose a review queue for this case.",
                    criteria={"standard": "Routine review", "urgent": "Urgent investigation"},
                )
            },
        )
        decision = classifier.invoke(request, config)
        return {"queue": decision.choices["queue"].choice}

    builder = StateGraph(ReviewState)
    builder.add_node("assess", assess)
    builder.add_edge(START, "assess")
    builder.add_edge("assess", END)
    graph = builder.compile()
    result = graph.invoke({"events": ["New device", "Payment to an unfamiliar recipient"]})
    print("Review queue:", result["queue"])
```

The graph invokes one decision; it does not perform an investigation or take
payment actions. In a real application, validate the policy and performance on
representative data, handle provider failures, and retain appropriate human review
and authorization. Send only the necessary evidence; enabled traces can contain
request data.

Next: [LangChain decisions](langchain-decisions.md), [errors and retries](../usage.md#errors-and-retries),
and [evaluation limitations](../evaluation.md).
