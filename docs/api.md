# API reference

## Classifier

::: langchain_finic.classifier.FinicClassifier
    options:
      members: [model, api_key, timeout, max_attempts, client, http_client, http_async_client, transport, async_transport, invoke, ainvoke, as_tool, close, aclose, to_json]

Inherited Runnable methods such as `batch`, `abatch`, `bind`, and `with_config`
follow the [LangChain Runnable interface](https://reference.langchain.com/python/langchain-core/runnables).
See the [usage guide](usage.md) for resource lifetime and invocation behavior.

## HTTP client and errors

::: langchain_finic.client.FinicClient
    options:
      members: [__init__, decide, adecide, score, ascore, create_score, acreate_score, list_scores, alist_scores, calibrate, acalibrate, get_job_status, aget_job_status, wait_for_job, await_job, close, aclose]

::: langchain_finic.client.FinicError

## Request and response types

::: langchain_finic.models
    options:
      members: [DecisionRequest, DecisionResponse, Noul, Choice, MultiChoice, Score, NoulAnswer, ChoiceAnswer, MultiChoiceAnswer, ScoreAnswer]

`Noul` and `Score` are public aliases. Their fields are defined by the following schemas:

::: langchain_finic.models.NoulQuestion

::: langchain_finic.models.ScoreQuestion

`Choice` and `MultiChoice` share the fields defined by `ChoiceQuestion`:

::: langchain_finic.models.ChoiceQuestion

## Optional middleware

Requires `langchain-finic[middleware]`.

::: langchain_finic.middleware
    options:
      members: [ModelChoice, ModelRouterMiddleware, AutoModeMiddleware, ModelRouterState]

## Custom scoring models

::: langchain_finic.scoring_models
    options:
      members: [ScoringInput, ScoringRequest, ScoringResponse, CreateScoreRequest, CreateScoreResponse, CalibrationRequest, CalibrationResponse, ScoreModel, ListScoresResponse, JobStatus, JobFailure]

## Scoring Runnable and toolkit

::: langchain_finic.scorer.FinicScorer
    options:
      members: [model_id, invoke, ainvoke, batch, abatch, as_tool]

::: langchain_finic.toolkit.FinicToolkit
    options:
      members: [client, operations, get_tools]
