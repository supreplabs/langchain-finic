"""Typed forwarding for Finic bindings; LangChain retains execution/config behavior."""

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Generic, Literal, Self, TypeVar, cast, overload

from langchain_core.runnables import RunnableBinding, RunnableConfig

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class FinicBinding(RunnableBinding[InputT, OutputT], Generic[InputT, OutputT]):
    """Retain dictionary inputs and exception-result types after binding."""

    def with_config(self, config: RunnableConfig | None = None, **kwargs: Any) -> Self:
        return cast(Self, super().with_config(config, **kwargs))

    def bind(self, **kwargs: Any) -> Self:
        return cast(Self, super().bind(**kwargs))

    def invoke(
        self, input: InputT | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> OutputT:
        return super().invoke(cast(InputT, input), config, **kwargs)

    async def ainvoke(
        self, input: InputT | dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> OutputT:
        return await super().ainvoke(cast(InputT, input), config, **kwargs)

    # LangChain omits Exception from batch results even when requested.
    @overload  # type: ignore[override]
    def batch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> list[OutputT]: ...

    @overload
    def batch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> list[OutputT | Exception]: ...

    @overload
    def batch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool,
        **kwargs: Any,
    ) -> list[OutputT | Exception]: ...

    def batch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[OutputT] | list[OutputT | Exception]:
        # Delegate unchanged to retain binding config merging and the bound runnable's
        # batching, cancellation and concurrency behavior. Invocation validates dicts.
        return super().batch(
            cast(list[InputT], inputs), config, return_exceptions=return_exceptions, **kwargs
        )

    # LangChain omits Exception from batch results even when requested.
    @overload  # type: ignore[override]
    async def abatch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> list[OutputT]: ...

    @overload
    async def abatch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> list[OutputT | Exception]: ...

    @overload
    async def abatch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool,
        **kwargs: Any,
    ) -> list[OutputT | Exception]: ...

    async def abatch(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[OutputT] | list[OutputT | Exception]:
        # Delegate unchanged to retain binding config merging and the bound runnable's
        # batching, cancellation and concurrency behavior. Invocation validates dicts.
        return await super().abatch(
            cast(list[InputT], inputs), config, return_exceptions=return_exceptions, **kwargs
        )

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, OutputT]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, OutputT | Exception]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool,
        **kwargs: Any,
    ) -> Iterator[tuple[int, OutputT | Exception]]: ...

    def batch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, OutputT | Exception]]:
        # Delegate unchanged to retain binding config merging and the bound runnable's
        # batching, cancellation and concurrency behavior. Invocation validates dicts.
        return super().batch_as_completed(
            cast(Sequence[InputT], inputs), config, return_exceptions=cast(Any, return_exceptions), **kwargs
        )

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, OutputT]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, OutputT | Exception]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, OutputT | Exception]]: ...

    def abatch_as_completed(
        self,
        inputs: Sequence[InputT | dict[str, Any]],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, OutputT | Exception]]:
        # Delegate unchanged to retain binding config merging and the bound runnable's
        # batching, cancellation and concurrency behavior. Invocation validates dicts.
        return super().abatch_as_completed(
            cast(Sequence[InputT], inputs), config, return_exceptions=cast(Any, return_exceptions), **kwargs
        )
