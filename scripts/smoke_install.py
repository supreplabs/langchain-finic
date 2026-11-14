"""Offline smoke check for an installed distribution; run outside the checkout."""

import argparse
import asyncio
import importlib.metadata
import importlib.util
import json

import httpx

from langchain_finic import DecisionRequest, FinicClassifier, Noul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--kairos", action="store_true")
    args = parser.parse_args()
    if args.base_only:
        assert importlib.util.find_spec("langchain") is None
        assert importlib.util.find_spec("langgraph") is None
        assert importlib.util.find_spec("kairos") is None
    else:
        from langchain_finic.middleware import AutoModeMiddleware, ModelRouterMiddleware

        assert AutoModeMiddleware and ModelRouterMiddleware

    def respond(request):
        assert request.headers["authorization"] == "Bearer synthetic-key"
        return httpx.Response(
            200,
            json={
                "id": "dec_smoke",
                "model": "grayson-1",
                "answers": {"risk": {"type": "noul", "noul": 0.3}},
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "request_id": "req_smoke",
                "evaluation_time_ms": 1,
            },
        )

    request = DecisionRequest(state="Fictional evidence", questions={"risk": Noul(instructions="Risk?")})
    with FinicClassifier(api_key="synthetic-key", transport=httpx.MockTransport(respond)) as classifier:
        assert classifier.invoke(request).nouls["risk"].noul == 0.3
        assert classifier.as_tool().invoke(request.model_dump())["request_id"] == "req_smoke"

    async def run():
        async with FinicClassifier(
            api_key="synthetic-key", async_transport=httpx.MockTransport(respond)
        ) as classifier:
            assert (await classifier.ainvoke(request)).request_id == "req_smoke"

    asyncio.run(run())

    if args.kairos:
        from kairos import Kairos

        from langchain_finic import KairosRunnable, KairosToolkit

        def kairos_transport(method, url, headers, body, timeout):
            assert method == "POST"
            assert url.endswith("/v1/ask")
            assert headers["Authorization"] == "Bearer synthetic-kairos-key"
            assert json.loads(body)["action"] == "synthetic_action"
            return 200, b'{"decision":"review","risk_score":55}', {}

        kairos_client = Kairos("synthetic-kairos-key", transport=kairos_transport)
        try:
            assert (
                KairosRunnable(client=kairos_client)
                .invoke({"action": "synthetic_action", "properties": {"user_id": "u1"}})
                .decision
                == "review"
            )
            assert [tool.name for tool in KairosToolkit(client=kairos_client).get_tools()] == [
                "kairos_decide"
            ]
        finally:
            kairos_client.shutdown()
        assert importlib.metadata.version("finic-kairos")

    print(f"Installed langchain-finic {importlib.metadata.version('langchain-finic')} passed smoke checks")


if __name__ == "__main__":
    main()
