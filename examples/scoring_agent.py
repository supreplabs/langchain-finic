"""Agent composition; install langchain-finic[middleware] and your chat provider.

Pass a tool-calling chat model and keep the FinicClient context open throughout
agent use. Default tools can score records and inspect models/jobs; they cannot
create or calibrate models. Supply stable, unique score IDs from the application.
"""

from langchain.agents import create_agent

from langchain_finic import FinicClient, FinicScorer, FinicToolkit


def build_agent(chat_model, client: FinicClient, model_id: str):
    scorer = FinicScorer(model_id=model_id, client=client)
    return create_agent(
        model=chat_model,
        tools=[scorer.as_tool(), *FinicToolkit(client=client).get_tools()],
        system_prompt=(
            "Use the provided score ID and evidence to assess the record. "
            "Scoring is a write: do not retry an ambiguous failure or invent a replacement score ID. "
            "Report the score and request ID; do not claim it proves fraud."
        ),
    )
