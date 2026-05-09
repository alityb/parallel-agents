from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


ToolFunc = Callable[..., Awaitable[Any]]


class ToolError(RuntimeError):
    """Tool execution failed in a way that should be reported to the agent."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    func: ToolFunc
    max_tokens: int = 2000
    cacheable: bool = True
    rate_limit: float | None = None
    key_arg: str | None = None
    batch_query: str | None = None
    cache_key_func: Callable[[dict[str, Any]], str] | None = None


class Tool:
    registry: dict[str, ToolDefinition] = {}

    @classmethod
    def define(
        cls,
        func: ToolFunc | None = None,
        *,
        name: str | None = None,
        max_tokens: int = 2000,
        cacheable: bool = True,
        rate_limit: float | None = None,
        cache_key_func: Callable[[dict[str, Any]], str] | None = None,
    ):
        def decorator(inner: ToolFunc) -> ToolDefinition:
            definition = ToolDefinition(
                name=name or inner.__name__,
                func=inner,
                max_tokens=max_tokens,
                cacheable=cacheable,
                rate_limit=rate_limit,
                key_arg=getattr(inner, "_batch_key_arg", None),
                batch_query=getattr(inner, "_batch_query", None),
                cache_key_func=cache_key_func,
            )
            cls.registry[definition.name] = definition
            setattr(cls, definition.name, definition)
            return definition

        if func is not None:
            return decorator(func)
        return decorator

    @staticmethod
    def batchable(*, key_arg: str, batch_query: str):
        def decorator(func: ToolFunc) -> ToolFunc:
            setattr(func, "_batch_key_arg", key_arg)
            setattr(func, "_batch_query", batch_query)
            return func

        return decorator


from . import builtin as _builtin  # noqa: E402,F401

__all__ = ["Tool", "ToolDefinition", "ToolError"]
