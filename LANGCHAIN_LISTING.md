# LangChain directory listing

Submit only after PyPI publication. This document prepares the listing; it does not
claim acceptance, partnership, or endorsement.

Use the [LangChain integration listing process](https://docs.langchain.com/oss/python/contributing/publish-langchain).
New packages normally receive an external-docs listing rather than a hosted guide.
Do not open a manual documentation PR unless a maintainer asks.

| Field | Value |
| --- | --- |
| Display name | Finic |
| Language | Python |
| Package | langchain-finic |
| Component | Tools; middleware (select the form's corresponding category) |
| Documentation | https://supreplabs.github.io/langchain-finic/ |
| Repository | https://github.com/supreplabs/langchain-finic |
| Description | Typed Finic Grayson decisions as LangChain tools and Runnables, with optional model routing and tool-screening middleware. |

`FinicClassifier.as_tool()` produces a standard LangChain tool. Middleware classes
extend `AgentMiddleware`. The integration is not a chat model; do not advertise
chat generation, token streaming, or provider-native tool calling.
