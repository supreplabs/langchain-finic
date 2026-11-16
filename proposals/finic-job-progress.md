# Proposal: optional job progress metadata

Status: prepared for upstream discussion; **not sent or implemented by Finic**.
Tracking: [#109](https://github.com/supreplabs/langchain-finic/issues/109).
Last reference check: 2026-11-16, [get_job_status](https://docs.finic.ai/reference/get-job-status).

## Problem and current contract

Creation and calibration can take minutes. Applications can currently distinguish a
pending job from terminal success or failure, but cannot explain what is happening
inside a pending job. The current response has `job_id`, `complete`, and optional
`error`. `complete=true` can mean failure. Local elapsed time and polling frequency
are observations about a client, not provider progress or a completion estimate.

The public dataset recipes now record local wait duration, completed waits, and
raised exception types. A raised wait does not establish a failed remote job;
timeouts and transport errors also raise. No private model records or response
error messages are needed to describe this friction.

## Proposed additive response

Keep existing fields and their semantics. Consider these optional fields, with the
provider deciding final names and availability. This is an illustrative proposal,
not a response shape applications can rely on today:

```json
{
  "job_id": "job_example",
  "complete": false,
  "progress": {
    "stage": "fitting",
    "fraction": null,
    "updated_at": "2026-11-16T10:00:00Z",
    "estimated_remaining_seconds": null
  }
}
```

- `stage`: extensible string, for example queued, validating, generating_questions,
  fitting, or finalizing. Clients must handle unfamiliar stages without failure.
- `fraction`: optional finite number in [0,1], only if a meaningful denominator is
  known. Omit or use null during indeterminate work. Specify whether it is global
  or stage-relative; do not synthesize it from elapsed time or input row count.
- `updated_at`: optional UTC timestamp of the last provider observation. A stale
  timestamp does not itself mean the job failed.
- `estimated_remaining_seconds`: optional nonnegative estimate, absent/null if
  unreliable. Document it as advisory and revisable, not a deadline guarantee.

Progress must never override `complete` and `error`. Fraction 1 does not authorize
scoring before terminal success. Specify how retries/restarts can change stage or
fraction, and how terminal failures are represented. Keep model records, generated
question text, secrets and internal stack traces out of progress responses.

## Options and tradeoffs

1. **Stage only first**: small additive contract, useful feedback without suggesting
   numerical certainty. Recommended first step if provider instrumentation permits.
2. **Stage plus measured fraction and optional ETA**: richer UI, but requires clear
   denominator, restart and reliability semantics. Omit values when not meaningful.
3. **Push notifications/webhooks**: may reduce polling but require authentication,
   delivery/replay semantics and a larger integration. Separate future proposal.

No change to the documented polling cadence, endpoint rate accounting, or completion
semantics is requested as a prerequisite. Ask the provider whether metadata can be
added to the existing GET endpoint with no extra client calls.

## Adoption and validation checklist

- [x] Document current contract and a concrete proposal without inventing support.
- [x] Record client wait observations in trial evidence; no percentage/ETA inferred.
- [x] Keep the weekly job-page fingerprint monitor active and test additive-field drift.
- [ ] Provider confirms supported fields, semantics and documentation.
- [ ] Link provider documentation and a sanitized accepted response fixture in #109.
- [ ] Add optional typed progress fields; retain old-response compatibility and public names.
- [ ] Test absent/null fields, unknown stages, invalid ranges, successful/failed jobs,
      and sync/async parity. Confirm progress never changes completion semantics.
- [ ] Update the compatibility matrix and reviewed digest only after implementation,
      contract tests and an explicitly authorized live validation if needed.

The weekly **Finic compatibility** workflow already fingerprints the whole job page
and opens a tracking issue for changed documentation. A regression verifies that an
additive progress field is a changed contract. Review that finding alongside #109;
do not automatically accept new fields or refresh hashes to clear a report.

Issue #109 remains open until the provider-dependent adoption criteria are resolved.
This proposal does not claim to have contacted the provider or received agreement.
