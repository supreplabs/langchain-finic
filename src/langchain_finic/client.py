"""HTTP transport for Finic decisions, custom scoring, and model lifecycle APIs."""

import asyncio
import math
import random
import re
import time
from concurrent.futures import CancelledError
from threading import Event, Lock
from typing import NoReturn, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from langchain_finic._config import env_option
from langchain_finic.models import DecisionRequest, DecisionResponse
from langchain_finic.scoring_models import (
    AddQuestionsRequest,
    CalibrationRequest,
    CalibrationResponse,
    CreateScoreRequest,
    CreateScoreResponse,
    CreateScoringModelRequest,
    CreateScoringModelResponse,
    DeleteScoringModelResponse,
    JobStatus,
    ListScoresResponse,
    ListScoringModelsResponse,
    ScoringModel,
    ScoringRequest,
    ScoringResponse,
    UpdateScoringModelRequest,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)

RETRYABLE = {429, 500, 502, 503, 504}


class FinicError(RuntimeError):
    """Provider or response-contract error with optional HTTP status and request ID."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        request_id: str | None = None,
        error_type: str | None = None,
        param: str | None = None,
    ):
        self.error_type = error_type
        self.param = param
        self.status = status
        self.request_id = request_id
        super().__init__(message)


class ClientConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    api_key: SecretStr = Field(repr=False, exclude=True)
    timeout: float = Field(default=60, gt=30, le=180)
    max_attempts: int = Field(default=3, ge=1, le=5)

    @field_validator("api_key")
    @classmethod
    def nonempty_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("Set FINIC_API_KEY or pass api_key")
        return value


def _delay(attempt: int) -> float:
    return min(8, 0.5 * 2**attempt) * random.uniform(0.5, 1)


class FinicClient:
    """Synchronous/asynchronous Finic transport with bounded retries and owned-client cleanup.

    Configure credentials explicitly or through FINIC_API_KEY. Injected HTTP clients
    remain caller-owned. Use context managers for deterministic resource cleanup.
    """

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        *,
        timeout: float | None = None,
        max_attempts: int | None = None,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ):
        key = api_key if api_key is not None else env_option("API_KEY", "")
        self.config = ClientConfig(
            api_key=key if isinstance(key, SecretStr) else SecretStr(key),
            timeout=timeout if timeout is not None else env_option("TIMEOUT_SECONDS", 60),
            max_attempts=max_attempts if max_attempts is not None else env_option("MAX_ATTEMPTS", 3),
        )
        if http_client is not None and transport is not None:
            raise ValueError("Pass http_client or transport, not both")
        if http_async_client is not None and async_transport is not None:
            raise ValueError("Pass http_async_client or async_transport, not both")
        self._sync_client = http_client
        self._async_client = http_async_client
        self._owns_sync = http_client is None
        self._owns_async = http_async_client is None
        self._lock = Lock()
        self._closed = False
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self.transport = transport
        self.async_transport = async_transport

    def _options(self) -> dict:
        # Apply these per request even to borrowed clients. Never follow auth redirects.
        return {
            "headers": {
                "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            "timeout": self.config.timeout,
            "follow_redirects": False,
        }

    def _sync(self) -> httpx.Client:
        with self._lock:
            if self._closed:
                raise RuntimeError("FinicClient is closed")
            if self._sync_client is None:
                self._sync_client = httpx.Client(transport=self.transport)
            return self._sync_client

    def _async(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                raise RuntimeError("FinicClient is closed")
            if self._async_loop is not None and self._async_loop is not loop:
                raise RuntimeError("Use one event loop per FinicClient async lifetime")
            self._async_loop = loop
            if self._async_client is None:
                self._async_client = httpx.AsyncClient(transport=self.async_transport)
            return self._async_client

    def close(self) -> None:
        """Close owned synchronous resources. Use aclose after any async use."""
        with self._lock:
            if self._owns_async and self._async_client is not None and not self._async_client.is_closed:
                raise RuntimeError("Use await aclose() to close asynchronous resources")
            self._closed = True
            if self._owns_sync and self._sync_client is not None:
                self._sync_client.close()

    async def aclose(self) -> None:
        """Close owned resources; injected httpx clients remain caller-owned."""
        with self._lock:
            self._closed = True
        if self._owns_async and self._async_client is not None:
            await self._async_client.aclose()
        if self._owns_sync and self._sync_client is not None:
            self._sync_client.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("FinicClient is closed")
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        await self.aclose()

    @staticmethod
    def _body(request: BaseModel) -> bytes:
        body = request.model_dump_json(exclude_none=True).encode()
        if len(body) > 2_000_000:
            raise FinicError("Request exceeds Finic's 2 MB body limit; shorten the case")
        return body

    @staticmethod
    def _parse(response: httpx.Response, request: DecisionRequest) -> DecisionResponse:
        request_id = response.headers.get("x-request-id")
        if response.is_success:
            try:
                decision = DecisionResponse.model_validate(response.json())
                decision.validate_for(request)
                return decision
            except (ValidationError, ValueError):
                raise FinicError(
                    "Finic response does not match the decision contract", request_id=request_id
                ) from None
        FinicClient._raise_http_error(response)

    @staticmethod
    def _raise_http_error(response: httpx.Response) -> NoReturn:
        request_id = response.headers.get("x-request-id")
        try:
            payload = response.json()
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            error = error if isinstance(error, dict) else {}
        except ValueError:
            error = {}

        # Accept only bounded machine identifiers, never the provider message/body.
        def identifier(value):
            return (
                value
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,200}", value)
                else None
            )

        hints = {
            400: "Invalid Finic request",
            401: "Check FINIC_API_KEY; authentication failed",
            402: "Finic credits exhausted; check billing in the portal",
            413: "Case exceeds Finic's token or body size limit",
            429: "Finic rate limit reached",
        }
        # Upstream bodies can contain submitted case data. Do not echo them.
        raise FinicError(
            hints.get(response.status_code, "Finic request failed") + f" (HTTP {response.status_code})",
            status=response.status_code,
            request_id=request_id,
            error_type=identifier(error.get("type")),
            param=identifier(error.get("param")),
        )

    def _send(
        self,
        method: str,
        path: str,
        *,
        request: BaseModel | None = None,
        retry: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        body = self._body(request) if request is not None else None
        client = self._sync()
        attempts = self.config.max_attempts if retry else 1
        options = self._options()
        if timeout is not None:
            options["timeout"] = min(timeout, self.config.timeout)
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                response = client.request(method, "https://api.finic.ai" + path, content=body, **options)
            except httpx.TransportError:
                if last:
                    raise FinicError(
                        "Finic transport failed; write outcome may be unknown"
                        if method == "POST" and not retry
                        else "Finic unreachable after bounded retries"
                    ) from None
            else:
                if response.status_code not in RETRYABLE or last:
                    return response
            time.sleep(_delay(attempt))
        raise FinicError("Finic request did not complete")

    async def _asend(
        self,
        method: str,
        path: str,
        *,
        request: BaseModel | None = None,
        retry: bool = False,
        timeout: float | None = None,
    ) -> httpx.Response:
        body = self._body(request) if request is not None else None
        client = self._async()
        attempts = self.config.max_attempts if retry else 1
        options = self._options()
        if timeout is not None:
            options["timeout"] = min(timeout, self.config.timeout)
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                response = await client.request(
                    method, "https://api.finic.ai" + path, content=body, **options
                )
            except httpx.TransportError:
                if last:
                    raise FinicError(
                        "Finic transport failed; write outcome may be unknown"
                        if method == "POST" and not retry
                        else "Finic unreachable after bounded retries"
                    ) from None
            else:
                if response.status_code not in RETRYABLE or last:
                    return response
            await asyncio.sleep(_delay(attempt))
        raise FinicError("Finic request did not complete")

    @staticmethod
    def _result(response: httpx.Response, schema: type[ResponseT]) -> ResponseT:
        if not response.is_success:
            FinicClient._raise_http_error(response)
        try:
            return schema.model_validate(response.json())
        except (ValueError, ValidationError):
            raise FinicError(
                "Finic response does not match the expected contract",
                request_id=response.headers.get("x-request-id"),
            ) from None

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        return self._parse(self._send("POST", "/v1/decide", request=request, retry=True), request)

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self._parse(await self._asend("POST", "/v1/decide", request=request, retry=True), request)

    def ask(self, request: DecisionRequest) -> DecisionResponse:
        """Ask the current Grayson API; equivalent to the retained ``decide`` alias."""
        response = self._send("POST", "/v1/ask", request=request, retry=True)
        result = self._parse(response, request)
        request_id = response.headers.get("x-request-id")
        if (
            result.request_id is None
            or result.evaluation_time_ms is None
            or (request_id is not None and result.request_id != request_id)
        ):
            raise FinicError(
                "Finic ask response does not match the current response contract", request_id=request_id
            )
        return result

    async def aask(self, request: DecisionRequest) -> DecisionResponse:
        response = await self._asend("POST", "/v1/ask", request=request, retry=True)
        result = self._parse(response, request)
        request_id = response.headers.get("x-request-id")
        if (
            result.request_id is None
            or result.evaluation_time_ms is None
            or (request_id is not None and result.request_id != request_id)
        ):
            raise FinicError(
                "Finic ask response does not match the current response contract", request_id=request_id
            )
        return result

    @staticmethod
    def _scored(response: httpx.Response, request: ScoringRequest) -> ScoringResponse:
        result = FinicClient._result(response, ScoringResponse)
        try:
            result.validate_for(request)
        except ValueError:
            raise FinicError(
                "Finic scoring response identity mismatch", request_id=response.headers.get("x-request-id")
            ) from None
        return result

    def score(self, request: ScoringRequest) -> ScoringResponse:
        """Score once. Never replay an ambiguous write or generate a score ID."""
        return self._scored(self._send("POST", "/v1/score", request=request), request)

    async def ascore(self, request: ScoringRequest) -> ScoringResponse:
        return self._scored(await self._asend("POST", "/v1/score", request=request), request)

    @staticmethod
    def _model_path(model_id: str) -> str:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be nonblank text")
        return "/v1/scoring_models/" + quote(model_id, safe="")

    @staticmethod
    def _model_result(response: httpx.Response, model_id: str) -> ScoringModel:
        model = FinicClient._result(response, ScoringModel)
        if model.model_id != model_id:
            raise FinicError(
                "Finic scoring model response identity mismatch",
                request_id=response.headers.get("x-request-id"),
            )
        return model

    def create_scoring_model(self, request: CreateScoringModelRequest) -> CreateScoringModelResponse:
        """Create a scoring model from labeled data or criteria; accepted jobs are not polled."""
        result = self._result(
            self._send("POST", "/v1/scoring_models", request=request), CreateScoringModelResponse
        )
        self._validate_creation_mode(request, result)
        return result

    async def acreate_scoring_model(self, request: CreateScoringModelRequest) -> CreateScoringModelResponse:
        result = self._result(
            await self._asend("POST", "/v1/scoring_models", request=request), CreateScoringModelResponse
        )
        self._validate_creation_mode(request, result)
        return result

    @staticmethod
    def _validate_creation_mode(
        request: CreateScoringModelRequest, result: CreateScoringModelResponse
    ) -> None:
        if (request.data is None) == (result.job_id is not None):
            raise FinicError("Finic creation response does not match the requested creation mode")

    def list_scoring_models(self) -> ListScoringModelsResponse:
        """List up to 500 current scoring models, newest first."""
        return self._result(self._send("GET", "/v1/scoring_models", retry=True), ListScoringModelsResponse)

    async def alist_scoring_models(self) -> ListScoringModelsResponse:
        return self._result(
            await self._asend("GET", "/v1/scoring_models", retry=True), ListScoringModelsResponse
        )

    def get_scoring_model(self, model_id: str) -> ScoringModel:
        return self._model_result(self._send("GET", self._model_path(model_id), retry=True), model_id)

    async def aget_scoring_model(self, model_id: str) -> ScoringModel:
        return self._model_result(await self._asend("GET", self._model_path(model_id), retry=True), model_id)

    def update_scoring_model(self, model_id: str, request: UpdateScoringModelRequest) -> ScoringModel:
        return self._model_result(
            self._send("PATCH", self._model_path(model_id), request=request, retry=True), model_id
        )

    async def aupdate_scoring_model(self, model_id: str, request: UpdateScoringModelRequest) -> ScoringModel:
        return self._model_result(
            await self._asend("PATCH", self._model_path(model_id), request=request, retry=True), model_id
        )

    def add_questions(self, model_id: str, request: AddQuestionsRequest) -> ScoringModel:
        return self._model_result(
            self._send("POST", self._model_path(model_id) + "/questions", request=request), model_id
        )

    async def aadd_questions(self, model_id: str, request: AddQuestionsRequest) -> ScoringModel:
        return self._model_result(
            await self._asend("POST", self._model_path(model_id) + "/questions", request=request), model_id
        )

    def remove_question(self, model_id: str, question_id: str) -> ScoringModel:
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question_id must be nonblank text")
        path = self._model_path(model_id) + "/questions/" + quote(question_id, safe="")
        return self._model_result(self._send("DELETE", path), model_id)

    async def aremove_question(self, model_id: str, question_id: str) -> ScoringModel:
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("question_id must be nonblank text")
        path = self._model_path(model_id) + "/questions/" + quote(question_id, safe="")
        return self._model_result(await self._asend("DELETE", path), model_id)

    def delete_scoring_model(self, model_id: str) -> DeleteScoringModelResponse:
        """Permanently delete a model and its records; archive it to disable reversibly."""
        result = self._result(self._send("DELETE", self._model_path(model_id)), DeleteScoringModelResponse)
        if result.model_id != model_id:
            raise FinicError("Finic deletion response identity mismatch")
        return result

    async def adelete_scoring_model(self, model_id: str) -> DeleteScoringModelResponse:
        result = self._result(
            await self._asend("DELETE", self._model_path(model_id)), DeleteScoringModelResponse
        )
        if result.model_id != model_id:
            raise FinicError("Finic deletion response identity mismatch")
        return result

    def create_score(self, request: CreateScoreRequest) -> CreateScoreResponse:
        """Start creation once; return the model/job IDs without polling."""
        return self._result(self._send("POST", "/v1/scores", request=request), CreateScoreResponse)

    async def acreate_score(self, request: CreateScoreRequest) -> CreateScoreResponse:
        return self._result(await self._asend("POST", "/v1/scores", request=request), CreateScoreResponse)

    def list_scores(self) -> ListScoresResponse:
        """List up to 500 scores, newest first (the API has no pagination)."""
        return self._result(self._send("GET", "/v1/scores", retry=True), ListScoresResponse)

    async def alist_scores(self) -> ListScoresResponse:
        return self._result(await self._asend("GET", "/v1/scores", retry=True), ListScoresResponse)

    def calibrate(self, request: CalibrationRequest) -> CalibrationResponse:
        """Submit outcomes once; return the accepted job without polling."""
        return self._result(self._send("POST", "/v1/calibrate", request=request), CalibrationResponse)

    async def acalibrate(self, request: CalibrationRequest) -> CalibrationResponse:
        return self._result(await self._asend("POST", "/v1/calibrate", request=request), CalibrationResponse)

    @staticmethod
    def _job_path(job_id: str) -> str:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be nonblank text")
        return "/v1/jobs/" + quote(job_id, safe="")

    @staticmethod
    def _job(response: httpx.Response, job_id: str) -> JobStatus:
        job = FinicClient._result(response, JobStatus)
        if job.job_id != job_id:
            raise FinicError(
                "Finic job response identity mismatch", request_id=response.headers.get("x-request-id")
            )
        return job

    def get_job_status(self, job_id: str) -> JobStatus:
        return self._job(self._send("GET", self._job_path(job_id), retry=True), job_id)

    async def aget_job_status(self, job_id: str) -> JobStatus:
        return self._job(await self._asend("GET", self._job_path(job_id), retry=True), job_id)

    @staticmethod
    def _poll_options(timeout: float, poll_interval: float) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (timeout, poll_interval)):
            raise ValueError("timeout and poll_interval must be positive finite seconds")

    @staticmethod
    def _completed(job: JobStatus) -> JobStatus:
        if job.error is not None:
            raise FinicError("Finic job failed", error_type=job.error.type)
        return job

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float = 900,
        poll_interval: float = 30,
        cancel_event: Event | None = None,
    ) -> JobStatus:
        """Poll to completion; cancellation is checked between requests and during waits.

        The deadline prevents new polls and bounds HTTP phase timeouts. An in-flight
        synchronous request cannot be forcibly interrupted by the cancellation event.
        """
        self._poll_options(timeout, poll_interval)
        path = self._job_path(job_id)
        deadline = time.monotonic() + timeout
        event = cancel_event if cancel_event is not None else Event()
        while True:
            if event.is_set():
                raise CancelledError("Finic job wait cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Finic job wait deadline exceeded")
            job = self._job(self._send("GET", path, timeout=remaining), job_id)
            if event.is_set():
                raise CancelledError("Finic job wait cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Finic job wait deadline exceeded")
            if job.complete:
                return self._completed(job)
            event.wait(min(poll_interval, max(0, deadline - time.monotonic())))

    async def await_job(
        self,
        job_id: str,
        *,
        timeout: float = 900,
        poll_interval: float = 30,
    ) -> JobStatus:
        """Poll with an overall deadline; task cancellation stops pending I/O/sleep."""
        self._poll_options(timeout, poll_interval)
        path = self._job_path(job_id)
        async with asyncio.timeout(timeout):
            while True:
                job = self._job(await self._asend("GET", path), job_id)
                if job.complete:
                    return self._completed(job)
                await asyncio.sleep(poll_interval)
