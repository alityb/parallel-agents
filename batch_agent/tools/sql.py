"""SQL batch grouping — coalesces multiple single-key queries into one batched query.

Tools annotated with @Tool.batchable(key_arg="id", batch_query="SELECT ... WHERE id IN ({ids})")
can have their calls batched when multiple agents request the same query template
with different key values within a short window.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import Tool, ToolDefinition

logger = logging.getLogger(__name__)

# Batch window: how long to wait for more calls before executing the batch
BATCH_WINDOW_MS = 5


class BatchCollector:
    """Collects batchable tool calls and executes them as a single query."""

    def __init__(self) -> None:
        self._pending: dict[str, _BatchGroup] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def call_or_batch(self, definition: ToolDefinition, args: dict[str, Any]) -> Any:
        """If the tool is batchable, collect and batch. Otherwise delegate to normal call."""
        if not definition.key_arg or not definition.batch_query:
            # Not batchable — fall through to normal execution
            return await definition.func(**args)

        key_value = args.get(definition.key_arg)
        if key_value is None:
            return await definition.func(**args)

        group_key = f"{definition.name}:{definition.batch_query}"

        if group_key not in self._locks:
            self._locks[group_key] = asyncio.Lock()

        # Add to pending group
        if group_key not in self._pending:
            self._pending[group_key] = _BatchGroup(definition=definition, batch_query=definition.batch_query)

        group = self._pending[group_key]
        future = asyncio.get_running_loop().create_future()
        group.items.append(_BatchItem(key_value=key_value, args=args, future=future))

        # If this is the first item, schedule the batch execution
        if len(group.items) == 1:
            asyncio.get_running_loop().call_later(
                BATCH_WINDOW_MS / 1000.0,
                lambda gk=group_key: asyncio.ensure_future(self._flush(gk)),
            )

        return await future

    async def _flush(self, group_key: str) -> None:
        """Execute the batched query and distribute results."""
        if group_key not in self._pending:
            return

        group = self._pending.pop(group_key)
        items = group.items

        if not items:
            return

        if len(items) == 1:
            # Only one item — just call normally
            item = items[0]
            try:
                result = await group.definition.func(**item.args)
                item.future.set_result(result)
            except Exception as exc:
                item.future.set_exception(exc)
            return

        # Multiple items — batch them
        key_values = [item.key_value for item in items]
        logger.info("[batch] Batching %d calls to %s (keys: %s)",
                    len(items), group.definition.name, key_values[:5])

        try:
            # Call the batch function if available, otherwise call individually
            if hasattr(group.definition.func, '_batch_handler'):
                results = await group.definition.func._batch_handler(key_values)
                for item, result in zip(items, results):
                    item.future.set_result(result)
            else:
                # Fallback: execute individually (still deduplicated by the pool)
                tasks = [group.definition.func(**item.args) for item in items]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for item, result in zip(items, results):
                    if isinstance(result, Exception):
                        item.future.set_exception(result)
                    else:
                        item.future.set_result(result)
        except Exception as exc:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)


class _BatchGroup:
    def __init__(self, definition: ToolDefinition, batch_query: str) -> None:
        self.definition = definition
        self.batch_query = batch_query
        self.items: list[_BatchItem] = []


class _BatchItem:
    def __init__(self, key_value: Any, args: dict[str, Any], future: asyncio.Future) -> None:
        self.key_value = key_value
        self.args = args
        self.future = future
