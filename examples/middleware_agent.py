"""Compose a chat-model router and tool guard with Finic. Running this spends API credits.

Install your chosen LangChain chat-provider packages first, then supply their
credentials alongside FINIC_API_KEY in the environment. FINIC_MODEL
selects the decision model; --fast-model and --powerful-model select chat models.
"""

import argparse

from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_finic import FinicClassifier
from langchain_finic.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware


@tool
def lookup_demo_case(case_id: str) -> dict:
    """Return the single fictional demo case. This tool accesses no live customer data."""
    if case_id != "demo":
        return {"error": "Only the fictional demo case is available."}
    return {"case_id": "demo", "status": "awaiting independent vendor callback"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-model", required=True, help="Provider-qualified chat-model name")
    parser.add_argument("--powerful-model", required=True, help="Provider-qualified chat-model name")
    parser.add_argument("--query", default="Look up the demo case and explain its status.")
    args = parser.parse_args()
    with FinicClassifier() as classifier:
        router = ModelRouterMiddleware(
            classifier=classifier,
            choices={
                "fast": ModelChoice(model=args.fast_model, criteria="Simple lookup or extraction."),
                "powerful": ModelChoice(model=args.powerful_model, criteria="Complex evidence analysis."),
            },
            minimum_probability=0.7,
            fallback="powerful",
        )
        guard = AutoModeMiddleware(classifier=classifier, tools=[lookup_demo_case])
        agent = create_agent(args.fast_model, tools=[lookup_demo_case], middleware=[router, guard])
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.query}]}, config={"recursion_limit": 10}
        )
        print(result["finic_model_route"])
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
