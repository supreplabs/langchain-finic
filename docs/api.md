---
description: API reference for Finic decision, scoring, and middleware integrations in Python.
---

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
      members: [__init__, decide, adecide, score, ascore, create_scoring_model, acreate_scoring_model, list_scoring_models, alist_scoring_models, get_scoring_model, aget_scoring_model, update_scoring_model, aupdate_scoring_model, delete_scoring_model, adelete_scoring_model, add_questions, aadd_questions, remove_question, aremove_question, calibrate, acalibrate, get_job_status, aget_job_status, wait_for_job, await_job, close, aclose]

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
      members: [ScoringInput, ScoringRequest, ScoringResponse, ScoredQuestion, CreateScoringModelRequest, CreateScoringModelResponse, UpdateScoringModelRequest, DeleteScoringModelResponse, AddQuestionsRequest, AddQuestionsResponse, AddedQuestion, CalibrationRequest, CalibrationResponse, ScoringModel, ScoringModelDetail, ScoringModelQuestion, ListScoringModelsResponse, JobStatus, JobFailure]

## Scoring Runnable and toolkit

::: langchain_finic.scorer.FinicScorer
    options:
      members: [model_id, invoke, ainvoke, batch, abatch, as_tool]

::: langchain_finic.toolkit.FinicToolkit
    options:
      members: [client, operations, get_tools]

## Kairos Runnable and toolkit

Requires `langchain-finic[kairos]` for the `finic-kairos` client.

::: langchain_finic.kairos
    options:
      members: [KairosDecisionInput, KairosDecisionOutput, KairosEventInput, KairosTrackAndDecideInput, KairosWorkflowInput, KairosRunnable, KairosToolkit]
