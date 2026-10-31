---
description: Test a langchain-finic application with a fixed fictional HTTP response and no provider credentials.
---

# Testing an application offline

Use HTTPX `MockTransport` to exercise a consumer with fixed fictional responses.
This complete example runs with the base package and makes one mocked request;
it requires no Finic account or provider credentials.

```python
import json

import httpx

from langchain_finic import DecisionRequest, FinicClassifier, Noul

fixture_calls = []


def fixture_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    fixture_calls.append(payload)
    assert request.url.path == "/v1/decide"
    return httpx.Response(
        200,
        json={
            "id": "fictional-decision",
            "model": payload["model"],
            "answers": {"urgent": {"type": "noul", "noul": 0.1}},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "request_id": "fictional-request",
            "evaluation_time_ms": 1,
        },
    )


request = DecisionRequest(
    data={"ticket": "Fictional support question"},
    questions={"urgent": Noul(instructions="Does this ticket require urgent review?")},
)
with FinicClassifier(
    api_key="offline-example-only",
    transport=httpx.MockTransport(fixture_response),
    max_attempts=1,
) as classifier:
    result = classifier.invoke(request)
    assert result.nouls["urgent"].noul == 0.1

assert len(fixture_calls) == 1
```

The transport supplies the answer and usage values. These checks establish request
construction, response parsing and the consumer's handling of that fixed result.
Provider authentication, availability, prediction quality, latency and billing
need separate live evidence.

Use a fresh fixture for each scenario and add application assertions about the
resulting queue, tool effect or fallback. Owned transports close with their
classifier; see [client ownership](usage.md#client-ownership) when sharing clients.
For native tools, preserve the question tags described in
[typed request conversion](usage.md#building-native-tool-arguments-from-a-typed-request).
