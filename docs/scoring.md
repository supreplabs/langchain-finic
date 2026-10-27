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
  100 labeled examples, with at least 20 of each outcome. Labels are booleans,
  integer 0/1, or case-insensitive text true/false/1/0/yes/no in either JSON or CSV.
  The original label representation is preserved in the request. Each example needs another field. Finic samples more than 2,000 rows;
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
