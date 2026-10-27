# Failure-contract coverage

This map supports the beta-readiness review in [issue #8](https://github.com/supreplabs/langchain-finic/issues/8).
These checks run offline with synthetic HTTP responses; provider outages and throttling
are not deliberately induced by live tests.

| Contract | Coverage |
| --- | --- |
| Permanent HTTP errors do not retry or echo response bodies | `unit_tests/test_classifier.py::test_permanent_errors_do_not_retry` |
| 429 and retryable 5xx responses recover within the attempt budget | `unit_tests/test_classifier.py::test_transient_errors_retry`, `test_async_retry_and_error` |
| Read timeouts and exhausted 429/503 retries preserve error metadata and close owned transports | `unit_tests/test_failure_contracts.py::test_failure_budget_redaction_and_cleanup` (sync/async) |
| Invalid JSON fails immediately without exposing the upstream body | `unit_tests/test_failure_contracts.py::test_failure_budget_redaction_and_cleanup` |
| Missing answers and inconsistent probability distributions fail validation | `unit_tests/test_classifier.py::test_malformed_response_fails_closed`, `test_inconsistent_distribution_rejected`, `test_bad_probabilities_rejected` |
| Partial batches preserve input order with `return_exceptions=True`; otherwise raise | `unit_tests/test_failure_contracts.py::test_partial_batch_preserves_order_or_raises` (sync/async) |
| Cancellation propagates during requests and retry backoff | `unit_tests/test_integration.py::test_cancellation_does_not_retry`, `unit_tests/test_failure_contracts.py::test_cancellation_during_backoff_stops_requests_and_closes` |
| Configurable invocation pools close after success, failure, and cancellation | `unit_tests/test_integration.py::test_configurable_sync_pools_closed_per_invocation`, `test_configurable_async_pools_closed` |
| Injected clients remain caller-owned | `unit_tests/test_integration.py::test_configured_models_leave_borrowed_http_pools_open`, `test_injected_client_ownership_and_headers` |
| Middleware fails closed or uses an explicit fallback | `unit_tests/test_middleware.py::test_guard_fail_closed_and_redacts_provider_errors`, `test_router_low_probability_and_error_fallback` |

## Intentional limits

Retries may repeat billable decisions. `max_attempts=1` disables retries; application
and LangGraph retry policies can multiply the total attempt count. Batches issue
individual HTTP requests and are not atomic: raising from a batch does not roll back
requests already sent. Cancellation stops local work but cannot retract a request
already received by Finic. The client uses bounded jittered backoff; it does not
currently interpret `Retry-After` headers.

These tests establish transport and framework behavior, not decision accuracy.
