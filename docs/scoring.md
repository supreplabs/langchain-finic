# Custom scores and model lifecycle

`FinicClient` supports all six documented Finic REST operations. `Score` remains
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

`score`/`ascore` return an integer from 0 to 1000. The client checks both model
and record identity in the response. `skip_schema_validation` and `update_schema`
are explicit flags: the latter changes the remote model's schema. Default
validation against the stored schema remains Finic's responsibility.

## Creating, discovering, and calibrating

- `create_score(CreateScoreRequest(data=examples, description=...))` starts model
  creation and returns model, job, and request IDs immediately. Supply at least
  100 labeled examples, with at least 20 of each outcome. Labels are booleans or
  integer 0/1. Each example needs another field. Finic samples more than 2,000 rows;
  the client never silently truncates your data.
- `list_scores()` returns the newest 500 models and their status/count metadata.
  Finic does not document pagination for this endpoint.
- `calibrate(CalibrationRequest(model_id=..., data=outcomes))` starts a calibration
  job. Outcomes contain a previously submitted `score_id` and `is_bad`; a row's
  optional `model_id` must match. Repeated outcomes replace existing labels.
- `get_job_status(job_id)` returns `complete`, an optional typed `error`, and a
  derived `succeeded` property. **Complete does not imply success.**

All five methods have native async equivalents prefixed with `a`. Use
`async with FinicClient()` for async work. Injected httpx clients remain
caller-owned; close them separately. These methods use the same `FINIC_API_KEY`
configuration as decisions.

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
        client=client, operations=("create_score",),
    ).get_tools()[0]
    return read_tools, creation_tool
```

The defaults are `finic_list_scores` and `finic_get_job_status`. Select
`create_score` or `calibrate` explicitly when your application authorizes those
writes. All tools have Pydantic argument schemas and native async implementations.
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
