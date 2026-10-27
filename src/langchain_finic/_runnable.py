"""Private shared configuration and resource ownership for Finic Runnables."""

from typing import Any, Generic, Self, TypeVar

import httpx
from langchain_core.load.serializable import (
    SerializedConstructor,
    SerializedNotImplemented,
    to_json_not_implemented,
)
from langchain_core.runnables import RunnableSerializable
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, model_validator
from pydantic.json_schema import SkipJsonSchema

from langchain_finic.client import FinicClient

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class FinicRunnable(RunnableSerializable[InputT, OutputT], Generic[InputT, OutputT]):
    """Shared internals; concrete classes retain their public serialization IDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", hide_input_in_errors=True)
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    timeout: float | None = Field(default=None, gt=30, le=180)
    max_attempts: int | None = Field(default=None, ge=1, le=5)
    client: SkipJsonSchema[FinicClient | None] = Field(default=None, exclude=True, repr=False)
    http_client: SkipJsonSchema[httpx.Client | None] = Field(default=None, exclude=True, repr=False)
    http_async_client: SkipJsonSchema[httpx.AsyncClient | None] = Field(
        default=None, exclude=True, repr=False
    )
    transport: SkipJsonSchema[httpx.BaseTransport | None] = Field(default=None, exclude=True, repr=False)
    async_transport: SkipJsonSchema[httpx.AsyncBaseTransport | None] = Field(
        default=None, exclude=True, repr=False
    )
    _runtime_client: FinicClient = PrivateAttr()
    _closed: bool = PrivateAttr(default=False)
    _pool_connections: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def configure_client(self) -> Self:
        if self.client is not None:
            if any(
                value is not None
                for value in (
                    self.api_key,
                    self.timeout,
                    self.max_attempts,
                    self.http_client,
                    self.http_async_client,
                    self.transport,
                    self.async_transport,
                )
            ):
                raise ValueError("Pass either client or client options")
            self._runtime_client = self.client
        else:
            self._runtime_client = self._new_client()
            self.api_key = self._runtime_client.config.api_key
            self.timeout = self._runtime_client.config.timeout
            self.max_attempts = self._runtime_client.config.max_attempts
        return self

    def _new_client(self) -> FinicClient:
        return FinicClient(
            api_key=self.api_key,
            timeout=self.timeout,
            max_attempts=self.max_attempts,
            http_client=self.http_client,
            http_async_client=self.http_async_client,
            transport=self.transport,
            async_transport=self.async_transport,
        )

    @property
    def _client(self) -> FinicClient:
        return self._runtime_client

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "FINIC_API_KEY"}

    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        # Recreating borrowed clients/transports from JSON would silently change behavior.
        if any(
            value is not None
            for value in (
                self.client,
                self.http_client,
                self.http_async_client,
                self.transport,
                self.async_transport,
            )
        ):
            return to_json_not_implemented(self)
        return super().to_json()

    def close(self) -> None:
        if self.client is None:
            self._client.close()
        self._closed = True

    async def aclose(self) -> None:
        if self.client is None:
            await self._client.aclose()
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")
        self._pool_connections = True
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
