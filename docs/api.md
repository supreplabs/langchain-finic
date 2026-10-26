# API reference

## Classifier

::: langchain_finic.classifier.FinicClassifier

## HTTP client and errors

::: langchain_finic.client.FinicClient

::: langchain_finic.client.FinicError

## Request and response types

::: langchain_finic.models
    options:
      members: [DecisionRequest, DecisionResponse, Noul, Choice, MultiChoice, Score, NoulAnswer, ChoiceAnswer, MultiChoiceAnswer, ScoreAnswer]

## Optional middleware

Requires `langchain-finic[middleware]`.

::: langchain_finic.middleware
    options:
      members: [ModelChoice, ModelRouterMiddleware, AutoModeMiddleware, ModelRouterState]
