"""HTTP transport for the Finic decision API."""

import asyncio
import random
import re
import time
from threading import Lock
from typing import NoReturn

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from langchain_finic._config import env_option
from langchain_finic.models import DecisionRequest, DecisionResponse

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
    def _body(request: DecisionRequest) -> bytes:
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

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._body(request)
        client = self._sync()
        for attempt in range(self.config.max_attempts):
            last = attempt == self.config.max_attempts - 1
            try:
                response = client.post("https://api.finic.ai/v1/decide", content=body, **self._options())
            except httpx.TransportError:
                if last:
                    raise FinicError("Finic unreachable after bounded retries") from None
            else:
                if response.status_code not in RETRYABLE or last:
                    return self._parse(response, request)
            time.sleep(_delay(attempt))
        raise FinicError("Finic request did not complete")

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._body(request)
        client = self._async()
        for attempt in range(self.config.max_attempts):
            last = attempt == self.config.max_attempts - 1
            try:
                response = await client.post(
                    "https://api.finic.ai/v1/decide", content=body, **self._options()
                )
            except httpx.TransportError:
                if last:
                    raise FinicError("Finic unreachable after bounded retries") from None
            else:
                if response.status_code not in RETRYABLE or last:
                    return self._parse(response, request)
            await asyncio.sleep(_delay(attempt))
        raise FinicError("Finic request did not complete")
