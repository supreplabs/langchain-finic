---
description: Use Finic custom scoring models with LangChain and manage their lifecycle in Python.
---

# Custom scores and model lifecycle

`FinicClient` supports the current Finic scoring-model and decision REST operations. `Score` remains
an ordered question for the decision API; custom model scoring uses
`ScoringRequest` and `ScoringResponse`.

```python
from langchain_finic import FinicClient, ScoringRequest

# Use a ready model you created in Finic. Each score_id must be unique in that model.
def score_payment(model_id: str, score_id: str):
    with FinicClient() as client:
        return client.score(ScoringRequest(
            model_id=model_id,
            score_id=score_id,
            data={"amount": 200, "new_device": True},
        ))
```

`score`/`ascore` return a `ScoringResponse` whose `score` field is an integer from
0 to 1000. The client checks both model and record identity in the response.
`skip_schema_validation` and `update_schema`
are explicit flags: the latter changes the remote model's schema. Default
validation against the stored schema remains Finic's responsibility.

## Creating, discovering, and calibrating

Use the current scoring-model endpoints for new applications:

```python
from langchain_finic import (
    AddQuestionsRequest,
    CreateScoringModelRequest,
    FinicClient,
    UpdateScoringModelRequest,
)

with FinicClient() as client:
    created = client.create_scoring_model(
        CreateScoringModelRequest(
            name="Payment fraud",
            criteria="Card payments. Bad means a confirmed fraud chargeback.",
        )
    )  # Criteria-only creation returns immediately without a job ID.
    model = client.get_scoring_model(created.model_id)
    client.add_questions(created.model_id, AddQuestionsRequest(questions=["Is this a new device?"]))
    client.update_scoring_model(created.model_id, UpdateScoringModelRequest(archived=True))
```

`list_scoring_models()` returns the current `scoring_models` response. `get_scoring_model()`
returns question IDs, sources and weights. `add_questions()` and `remove_question()`
manage user questions; a question currently used by a calibrated score must be removed
through a calibration request's `remove_questions`. `update_scoring_model()` renames,
changes criteria, and archives or unarchives a model. `delete_scoring_model()` is
permanent and deletes its records. Archive first when the goal is to disable scoring
reversibly.

Model creation from labeled records returns a job ID; poll it with `get_job_status()`
or `wait_for_job()`. Creation with labeled data already calibrates the model.
Later calibration refits it to additional labeled data. Creation from criteria alone returns a model ID immediately.
Calibration takes at least 100 labeled records, with at least 20 records in each
outcome class, and can add or remove questions in the same request. Finic uses up
to 2,000 supplied labeled rows for creation/calibration.

See the [disposable scoring tutorial](tutorials/disposable-scoring.md) for an
executable evaluation, recovery ledger, threshold guidance and verified cleanup.

- `calibrate(CalibrationRequest(model_id=..., data=examples))` starts a calibration
  job with labeled records. The earlier `examples` request field is accepted as an
  alias for `data`. Earlier package versions accepted per-score outcome rows keyed
  by `score_id`; the current Finic endpoint instead requires full labeled records
  and the same 100-row / 20-per-outcome minimum as model creation. Keep the source
  records used for scoring if you plan to calibrate; score IDs alone are not enough.
- `get_job_status(job_id)` returns `complete`, an optional typed `error`, and a
  derived `succeeded` property. **Complete does not imply success.**

All client methods have native async equivalents prefixed with `a`. Use
`async with FinicClient()` for async work. Injected httpx clients remain
caller-owned; close them separately. These methods use the same `FINIC_API_KEY`
configuration as decisions.

## Legacy model creation and listing

For new applications use `create_scoring_model()` and `list_scoring_models()`.
The older `create_score()` and `list_scores()` methods remain available on legacy
routes. Their label and response contracts differ from the current methods:

- `create_score(CreateScoreRequest(data=examples, description=...))` starts model
  creation and returns model, job, and request IDs immediately. Supply at least
  100 labeled examples, with at least 20 of each outcome. Labels are booleans,
  integer 0/1, or case-insensitive text true/false/1/0/yes/no in either JSON or CSV.
  The original label representation is preserved in the request. Each example needs another field. Finic samples more than 2,000 rows;
  the client never silently truncates your data.
- `list_scores()` returns the newest 500 models and their status/count metadata.
  Finic does not document pagination for this endpoint.

## JSON and CSV

Scoring accepts an object, a one-object list, or CSV with a header and one row.
Creation and calibration accept lists of objects or CSV strings. CSV travels as
text in the JSON `data` field; file uploads/multipart are not implemented.
The client validates CSV row widths, unique headers, labels, and required fields,
but leaves CSV type coercion and dotted-field expansion to Finic. This preserves
provider semantics (including ZIP codes with leading zeros).

JSON records are checked against the 60,000-byte compact JSON limit, excluding
`is_bad`. CSV record-size checks after provider coercion happen on the server.
All request envelopes must fit the 2 MB limit. Descriptions are at most 4,000
characters and score IDs at most 200 characters. Validation rejects nonfinite
JSON numbers. Remote errors include `FinicError.error_type`, `.param`, `.status`,
and `.request_id`; default messages omit the provider body.

## Retries and waiting

Reads and decisions use bounded retries. **Scoring, creation, and calibration
make one attempt**, even when `max_attempts` is larger. A timeout may mean the
write succeeded: the client reports failure and never fabricates a result or
replaces your ID. A repeated successful score ID yields `duplicate_score_id`,
without the original score. Handle that explicitly. Automatic retries around
these methods or tools (including Runnable `with_retry`) can replay writes.

`wait_for_job(job_id, timeout=900, poll_interval=30)` returns a successful job or
raises on job failure/deadline. Sync cancellation uses an optional
`threading.Event`; it interrupts sleeps and prevents later polls, but cannot
interrupt an in-flight synchronous HTTP request. Its deadline bounds poll
starts and HTTP phase timeouts, not a hard wall-clock limit for a slow response.
`await_job` has a hard overall asyncio deadline and propagates task cancellation.
Both helpers use one attempt per poll and surface transient HTTP errors to the
caller. Waiting never restarts creation or calibration.

The legacy `create_score()` and `list_scores()` methods remain available for existing
applications. New code should use `create_scoring_model()` and
`list_scoring_models()` for the current `/v1/scoring_models` routes. Creation,
calibration, model updates, question changes, and deletion affect server-side state;
only expose the corresponding tools when the application authorizes those operations.

See the [Finic API reference](https://docs.finic.ai/reference) for provider limits
and [API types](api.md) for method signatures.

## Native LangChain scoring

`FinicScorer(model_id=...)` implements the typed Runnable interface using
`ScoringInput` and `ScoringResponse`. It supports `invoke`, `ainvoke`, LCEL,
`batch`, `abatch`, callbacks, configurable fields, and `as_tool()`.

```python
from langchain_finic import FinicScorer

async def score_records(model_id: str, records: list[dict]):
    async with FinicScorer(model_id=model_id) as scorer:
        # Each input contains its own unique score_id and data.
        return await scorer.abatch(records, {"max_concurrency": 2})
```

Direct scorer batches default to five concurrent calls; `max_concurrency`
overrides this. Other LangChain compositions follow their own batching defaults,
so pass an explicit concurrency limit to the outer chain. Concurrency is not a
requests-per-second limiter: Finic's organization rate and quotas still apply.
Batch results keep input order. `return_exceptions=True` preserves per-record
failures; a batch is not atomic and cancellation cannot retract accepted writes.
Streaming yields one complete result. Do not put automatic retry wrappers around
write operations without considering duplicate IDs and ambiguous outcomes.

The scorer uses the classifier's ownership rules: invocation-scoped owned pools
unless used as a context manager; injected clients remain caller-owned. Default
serialization contains a secret reference rather than an API key. Injected
transports/clients are not serializable. The bound `model_id` is configurable;
individual record inputs cannot override it.

Callbacks and LangSmith traces include the operation and model ID; successful
outputs retain request and score IDs. Tracing is opt-in and record contents are
not automatically redacted. The scorer does not fabricate token usage metadata
for an endpoint that does not return it.

## Selected lifecycle tools

```python
from langchain_finic import FinicClient, FinicToolkit

# Keep this client open for the entire lifetime of the returned tools.
def lifecycle_tools(client: FinicClient):
    read_tools = FinicToolkit(client=client).get_tools()
    creation_tool = FinicToolkit(
        client=client, operations=("create_scoring_model",),
    ).get_tools()[0]
    return read_tools, creation_tool
```

The defaults are `finic_list_scoring_models` and `finic_get_job_status`. Add
`create_scoring_model`, `calibrate`, `update_scoring_model`, `add_questions`, or
`remove_question` only when your application authorizes those writes. The permanent
`delete_scoring_model` tool is never included by default. All tools have Pydantic
argument schemas and native async implementations. `FinicError` propagates with
its structured metadata; an application can decide how to handle it. Job-status
tool results include the provider's failure details, so apply your application's
usual trace-access controls.

The base install includes the scorer and toolkit. Install the `middleware` extra
for the LangChain agent and LangGraph examples:

- [Scoring with LCEL](https://github.com/supreplabs/langchain-finic/blob/main/examples/scoring.py)
- [Agent with selected tools](https://github.com/supreplabs/langchain-finic/blob/main/examples/scoring_agent.py)
- [Checkpointed lifecycle graph](https://github.com/supreplabs/langchain-finic/blob/main/examples/lifecycle_graph.py)

The graph separates each write from its following wait and retains job/model IDs
in checkpoints. Resume a failed wait using the same thread and a `None` input;
do not start the graph over. Use durable checkpoints for process restarts. Even
with checkpoints, a write accepted remotely before a connection failure may need
manual reconciliation. The graph intentionally has no automatic write retries.

### Scoring tool capabilities and bindings

`scorer.as_tool()` exposes record scoring only. Schema mutation and validation
bypass are rejected at runtime as well as omitted from the default argument schema.
To deliberately grant either capability, construct the tool with
`allow_schema_update=True` or `allow_schema_bypass=True`. Direct `FinicClient`
and `FinicScorer.invoke` inputs still support both provider flags. Existing agents
that used those flags must opt in when upgrading. Custom argument schemas do not
bypass the capability check.

`with_config(...).as_tool()`, `bind().as_tool()`, and configurable-field tools retain
the `finic_score` name and JSON output. Explicit names/descriptions are supported.
Choose capabilities when converting the runnable into a tool; configuring that
tool afterwards retains its capability policy.

`batch_as_completed` and `abatch_as_completed` use the same default concurrency
of five as ordinary scorer batches and return each original input index with its
result. An explicit `max_concurrency` overrides the default. Cancellation cannot
retract writes already accepted by Finic. For organization-wide request pacing,
coordinate all workers through a shared rate limiter or queue before invoking
scoring; a semaphore alone does not enforce requests per second. Keep one attempt
per write and reconcile ambiguous failures rather than automatically replaying them.


### Inspecting destructive tools

`FinicToolkit.destructive` lists selected operations that permanently delete a
model and its records. It is empty for the default toolkit. Applications can check
this tuple before exposing tools to an agent. It is informational metadata, not an
authorization boundary: other explicitly selected tools can also change remote
state. Keep permission checks in the application.

A transport timeout during model deletion or question removal reports that the
write outcome may be unknown. These calls are not automatically retried; reconcile
the remote state before deciding whether another request is appropriate.

## Calibration migration and score evidence

The `examples` alias only renames the `data` field; it does not look up scored records.
Replace old rows such as `{"score_id": "payment-1", "is_bad": true}` with original
features such as `{"amount": 200, "new_device": true, "is_bad": true}`. This applies
to JSON and CSV. Supply at least 100 full records, including 20 of each outcome.

Pass `include_questions=True` in `ScoringInput` or `ScoringRequest` to receive
`response.questions`. Each entry contains question text, probability and weight;
when evidence was not requested it may be absent. These fields are inspection
metadata; the package does not define a formula for reconstructing the final score.
Recalibration can change question IDs, membership, count and weights; preserve
evidence per response and never join questions by array position.

Toolkit scoring requires `allow_schema_update=True` or `allow_schema_bypass=True`
on `FinicToolkit` before its score tool may use the corresponding schema flag.
For agents restricted to one model, prefer `scorer.as_tool()`.

The lifecycle graph accepts separate `examples` for creation and
`calibration_examples` for later calibration. Its `record` input is a scorer input,
for example `{"score_id": "payment-1", "data": {"amount": 200}}`. Accumulate known
outcomes with their original features before starting this demonstration; it does
not wait for an outcome from the single score it creates.

Adding/removing questions and changing criteria alter remote model configuration.
Inspect the returned model and question IDs after updates. Removal of questions
used by a calibrated score requires calibration's `remove_questions`. The package
does not establish concurrent scoring availability or whether criteria edits
require recalibration; confirm these provider behaviors before scheduling changes.


## Job status and progress

The [current job reference](https://docs.finic.ai/reference/get-job-status) exposes
`job_id`, `complete` and optional `error`; it explicitly does not report progress.
`complete=True` with an error is failure. `wait_for_job()` and `await_job()` return
only on terminal success and raise on terminal failure or a wait error. A timeout
stops the local wait; it does not cancel the remote job or authorize replaying creation.

Use an indeterminate waiting indicator and label local elapsed time as elapsed time.
Do not show a computed percent or ETA from row counts, polls, or historical durations.
The disposable trial records observed wait durations separately from cleanup status.

An [upstream proposal](https://github.com/supreplabs/langchain-finic/blob/main/proposals/finic-job-progress.md)
is tracked in [#109](https://github.com/supreplabs/langchain-finic/issues/109).
Proposed fields are not supported API fields. The weekly compatibility check tracks
changes to the job reference; typed progress support will follow a confirmed contract.
