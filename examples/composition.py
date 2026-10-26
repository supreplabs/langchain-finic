"""LCEL and LangGraph composition. Running this makes two paid Finic calls.

Install langgraph>=1,<2 and export FINIC_API_KEY first. Only fictional data is sent.
"""

from typing import NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from langchain_finic import Choice, DecisionRequest, DecisionResponse, FinicClassifier


class State(TypedDict):
    request: DecisionRequest
    decision: NotRequired[DecisionResponse]


def main():
    request = DecisionRequest(
        state={"events": ["Fictional account changed device before a new payment"]},
        questions={
            "route": Choice(
                instructions="Choose a review queue.",
                criteria={"routine": "Routine review", "urgent": "Urgent investigation"},
            )
        },
    )
    with FinicClassifier() as classifier:
        route = classifier | (lambda decision: decision.choices["route"].choice)
        print(route.invoke(request))

        def decide(state: State, config: RunnableConfig):
            return {"decision": classifier.invoke(state["request"], config)}

        builder = StateGraph(State)
        builder.add_node("decide", decide)
        builder.add_edge(START, "decide")
        builder.add_edge("decide", END)
        graph = builder.compile()
        result = graph.invoke({"request": request})
        print(result["decision"].model_dump_json())


if __name__ == "__main__":
    main()
