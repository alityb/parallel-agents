from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

from . import Tool


def _read_file_cache_key(args: dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser().resolve()
    stat = path.stat()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


@Tool.define(max_tokens=8000, cacheable=True, cache_key_func=_read_file_cache_key)
async def read_file(path: str, encoding: str = "utf-8") -> str:
    file_path = Path(path).expanduser().resolve()
    return await asyncio.to_thread(file_path.read_text, encoding=encoding)


@Tool.define(max_tokens=4000, cacheable=True, rate_limit=10)
async def http_get(url: str, timeout: float = 15) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


@Tool.define(max_tokens=2000, cacheable=True, rate_limit=10)
async def web_search(query: str) -> str:
    raise NotImplementedError("web_search requires a user-provided search backend in Phase 0")


@Tool.define(max_tokens=2000, cacheable=False)
async def python_eval(code: str) -> str:
    if os.getenv("BATCH_AGENT_ENABLE_PYTHON_EVAL") != "1":
        raise RuntimeError("python_eval is disabled unless BATCH_AGENT_ENABLE_PYTHON_EVAL=1")
    allowed_builtins = {"abs": abs, "len": len, "max": max, "min": min, "sum": sum, "range": range}
    return repr(eval(code, {"__builtins__": allowed_builtins}, {}))
