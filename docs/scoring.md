---
description: Use Finic custom scoring models with LangChain and manage their lifecycle in Python.
---

# Custom scores and model lifecycle

`FinicClient` supports all eleven documented Finic REST operations. `Score` remains
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
`include_questions` returns the questions behind the score, each with Grayson's
probability for this record and its weight; the client rejects a response that
carries questions you did not ask for.

## Creating, discovering, and calibrating

- `create_scoring_model(CreateScoringModelRequest(name=..., criteria=...))` creates a
  scoring model from its criteria alone. It is ready at once and `job_id` is `None`,
  which the `ready` property reports. It has no questions until you add your own or
  calibrate it.
- `create_scoring_model(CreateScoringModelRequest(name=..., data=examples))` creates
  one from labeled records and returns a `job_id` to wait on. Supply at least 100
  labeled records, with at least 20 of each outcome. Labels are booleans, integer
  0/1, or case-insensitive text true/false/1/0/yes/no in either JSON or CSV. The
  original label representation is preserved in the request. Each record needs
  another field. Finic samples more than 2,000 rows; the client never silently
  truncates your data. `add_questions` carries up to 20 of your own questions.
- `list_scoring_models()` returns the newest 500 models with their name, status,
  criteria, timestamps and counts. Finic does not document pagination for this
  endpoint.
- `get_scoring_model(model_id)` returns one model with the questions it uses,
  strongest first, each with its `source` and `weight`. `user_questions` filters
  the ones you added. Question IDs come from here.
- `add_questions(model_id, AddQuestionsRequest(questions=[...]))` adds up to 20 of
  your own yes/no questions without calibrating, and returns the model with an
  `added` list saying which are used. Adding a question the model already has is
  rejected, so this is not safe to repeat blindly.
- `remove_question(model_id, question_id)` removes one question. A question the
  calibrated score uses is refused with `question_in_use`; remove that one through
  `calibrate`'s `remove_questions` instead.
- `update_scoring_model(model_id, UpdateScoringModelRequest(archived=True))`
  renames, re-states criteria, or archives and unarchives. Archiving stops scoring
  and calibration without deleting anything, and repeating it is safe.
- `delete_scoring_model(model_id)` deletes a model and its records. **This cannot be
  undone** and Finic accepts it only with an API key an organization admin created.
  Archive instead unless deletion is intended.
- `calibrate(CalibrationRequest(model_id=..., data=records))` starts a calibration
  job and returns `job_id` with the `rows` used. It takes labeled records in the
  same shape as creation, not previously scored IDs, and accepts `add_questions`
  and `remove_questions`.
- `get_job_status(job_id)` returns `complete`, an optional typed `error`, and a
  derived `succeeded` property. **Complete does not imply success.**

Every method has a native async equivalent prefixed with `a`. Use
`async with FinicClient()` for async work. Injected httpx clients remain
caller-owned; close them separately. These methods use the same `FINIC_API_KEY`
configuration as decisions.

Reads and the idempotent `update_scoring_model` retry on a retryable status.
Creation, scoring, question changes, calibration and deletion are sent once: a
transport failure on one reports that the outcome may be unknown rather than
replaying it.

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

Finic documents no model deletion endpoint. Creation and calibration change
server-side state. Keep job/model IDs in your application, and enable those
operations only where your application intends to perform them.

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

The defaults are `finic_list_scoring_models`, `finic_get_scoring_model` and
`finic_get_job_status`, which only read. Select `create_scoring_model`,
`add_questions`, `remove_question`, `update_scoring_model` or `calibrate`
explicitly when your application authorizes those writes. `delete_scoring_model`
is never a default because it cannot be undone; `FinicToolkit.destructive` names
the selected operations that are irreversible, so an application can refuse to
build an agent that holds one. All tools have Pydantic argument schemas and native async implementations.
Creation/calibration return job handles immediately. `FinicError` propagates with
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
