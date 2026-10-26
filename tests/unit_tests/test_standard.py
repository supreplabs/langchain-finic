"""Official tool contracts, executed against synthetic HTTP responses offline."""

import httpx
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests.tools import ToolsIntegrationTests
from langchain_tests.unit_tests.tools import ToolsUnitTests

from langchain_finic import FinicClassifier


def response(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "dec_standard",
            "model": "grayson-1",
            "answers": {"risk": {"type": "noul", "noul": 0.25}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
            "request_id": "req_standard",
            "evaluation_time_ms": 1.0,
        },
    )


class FinicToolContract:
    @property
    def tool_constructor(self) -> BaseTool:
        # Invocation-scoped clients close after each call. No open pools are retained.
        return FinicClassifier(
            api_key="test-only",
            model="grayson-1",
            max_attempts=1,
            transport=httpx.MockTransport(response),
            async_transport=httpx.MockTransport(response),
        ).as_tool()

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "state": {"events": ["Fictional account review"]},
            "questions": {"risk": {"type": "noul", "instructions": "Is this suspicious?"}},
        }


class TestFinicToolUnit(FinicToolContract, ToolsUnitTests):
    pass


class TestFinicToolComposition(FinicToolContract, ToolsIntegrationTests):
    """The standard integration contract also runs offline using mock HTTP."""
