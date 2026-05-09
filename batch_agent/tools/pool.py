from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from . import Tool, ToolDefinition
from .sql import BatchCollector
from ..utils import p50, p75, stable_hash as _stable_hash_util

logger = logging.getLogger(__name__)

_batch_collector = BatchCollector()


class ToolPool:
    def __init__(self, *, cache_ttl: float = 60, max_cache_entries: int = 1024) -> None:
        self.cache_ttl = cache_ttl
        self.max_cache_entries = max_cache_entries
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._limiters: dict[str, _TokenBucket] = {}
        # Per-tool P50/P75/P99 latency tracking for KVFlow Advisor
        self._latencies: dict[str, list[float]] = {}

    async def call(self, tool: str | ToolDefinition, args: dict[str, Any] | None = None) -> Any:
        args = args or {}
        definition = self._resolve(tool)
        key = self._key(definition, args)

        # Route batchable tools through BatchCollector (Drift 7)
        if definition.key_arg and definition.batch_query:
            return await _batch_collector.call_or_batch(definition, args)

        if definition.cacheable:
            cached = self._get_cached(key)
            if cached is not _MISSING:
                return cached

        if key in self._inflight:
            return await self._inflight[key]

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._inflight[key] = future
        try:
            await self._rate_limit(definition)
            t0 = time.monotonic()
            result = await definition.func(**args)
            elapsed = time.monotonic() - t0
            self._record_latency(definition.name, elapsed)
            result = self._truncate(result, definition.max_tokens)
            if definition.cacheable:
                self._set_cached(key, result)
            future.set_result(result)
            return result
        except Exception as exc:
            future.set_exception(exc)
            raise
        except BaseException as exc:
            # CancelledError (tool timeout via asyncio.wait_for) reaches here.
            # If we don't resolve the future, all concurrent waiters hang forever.
            if not future.done():
                if isinstance(exc, asyncio.CancelledError):
                    future.set_exception(RuntimeError("tool execution cancelled"))
                else:
                    future.set_exception(exc)
            raise
        finally:
            self._inflight.pop(key, None)

    def p75_latency(self, tool_name: str) -> float | None:
        """Return P75 latency for a tool, or None if insufficient data."""
        data = self._latencies.get(tool_name)
        return p75(data) if data and len(data) >= 2 else None

    def p50_latency(self, tool_name: str) -> float | None:
        """Return P50 latency for a tool, or None if no data."""
        data = self._latencies.get(tool_name)
        return p50(data) if data else None

    def _record_latency(self, tool_name: str, elapsed: float) -> None:
        lst = self._latencies.setdefault(tool_name, [])
        lst.append(elapsed)
        # Cap at 1000 samples to avoid unbounded memory
        if len(lst) > 1000:
            lst.pop(0)

    def _resolve(self, tool: str | ToolDefinition) -> ToolDefinition:
        if isinstance(tool, ToolDefinition):
            return tool
        return Tool.registry[tool]

    def _key(self, definition: ToolDefinition, args: dict[str, Any]) -> str:
        if definition.cache_key_func:
            payload = definition.cache_key_func(args)
        else:
            payload = _stable_hash_util(args)
        return f"{definition.name}:{payload}"

    def _get_cached(self, key: str) -> Any:
        item = self._cache.get(key)
        if item is None:
            return _MISSING
        expires_at, value = item
        if expires_at < time.monotonic():
            self._cache.pop(key, None)
            return _MISSING
        self._cache.move_to_end(key)
        return value

    def _set_cached(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic() + self.cache_ttl, value)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)

    async def _rate_limit(self, definition: ToolDefinition) -> None:
        if not definition.rate_limit:
            return
        limiter = self._limiters.setdefault(definition.name, _TokenBucket(definition.rate_limit))
        await limiter.acquire()

    def _truncate(self, result: Any, max_tokens: int) -> Any:
        if not isinstance(result, str):
            return result
        max_chars = max_tokens * 4
        if len(result) <= max_chars:
            return result
        omitted = (len(result) - max_chars + 3) // 4
        return result[:max_chars] + f"\n[TRUNCATED - {omitted} tokens omitted]"


class _TokenBucket:
    def __init__(self, rate: float) -> None:
        self.rate = rate
        self.capacity = max(1.0, rate)
        self.tokens = self.capacity
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.updated_at = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


class _Missing:
    pass


_MISSING = _Missing()
