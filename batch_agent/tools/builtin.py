from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from . import Tool, ToolError


def _read_file_cache_key(args: dict[str, Any]) -> str:
    path = Path(args["path"]).expanduser().resolve()
    stat = path.stat()
    return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"


@Tool.define(max_tokens=8000, cacheable=True, cache_key_func=_read_file_cache_key)
async def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read the contents of a file from disk."""
    file_path = Path(path).expanduser().resolve()
    return await asyncio.to_thread(file_path.read_text, encoding=encoding)


@Tool.define(max_tokens=4000, cacheable=True, rate_limit=10)
async def http_get(url: str, timeout: float = 15) -> str:
    """Fetch the contents of a URL via HTTP GET."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


@Tool.define(max_tokens=4000, cacheable=True, rate_limit=5)
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return a summary of results.

    Backends supported (checked in order):
      1. Brave Search API  — set BRAVE_SEARCH_API_KEY
      2. SerpAPI           — set SERPAPI_KEY or SERPAPI

    If neither key is set, raises RuntimeError with setup instructions.
    Results are formatted as: "Title\nURL\nSnippet\n---" blocks.
    """
    brave_key = os.getenv("BRAVE_SEARCH_API_KEY")
    serp_key = os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI")

    if brave_key:
        return await _brave_search(query, num_results, brave_key)
    if serp_key:
        return await _serpapi_search(query, num_results, serp_key)

    raise RuntimeError(
        "web_search requires a search API key. "
        "Set one of:\n"
        "  BRAVE_SEARCH_API_KEY  — https://api.search.brave.com/\n"
        "  SERPAPI_KEY or SERPAPI — https://serpapi.com/\n"
        "Alternatively, register a custom web_search tool with your preferred backend."
    )


async def _brave_search(query: str, num_results: int, api_key: str) -> str:
    """Search via Brave Search API."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": num_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        response.raise_for_status()
    data = response.json()
    results = data.get("web", {}).get("results", [])
    return _format_results(results, title_key="title", url_key="url", snippet_key="description")


async def _serpapi_search(query: str, num_results: int, api_key: str) -> str:
    """Search via SerpAPI (Google backend)."""
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://serpapi.com/search",
            params={"q": query, "num": num_results, "api_key": api_key, "engine": "google"},
        )
        response.raise_for_status()
    data = response.json()
    results = data.get("organic_results", [])
    return _format_results(results, title_key="title", url_key="link", snippet_key="snippet")


def _format_results(
    results: list[dict[str, Any]],
    title_key: str,
    url_key: str,
    snippet_key: str,
) -> str:
    if not results:
        return "No results found."
    parts: list[str] = []
    for r in results:
        title = r.get(title_key, "")
        url = r.get(url_key, "")
        snippet = r.get(snippet_key, "")
        parts.append(f"{title}\n{url}\n{snippet}")
    return "\n---\n".join(parts)


@Tool.define(max_tokens=2000, cacheable=False)
async def python_eval(code: str) -> str:
    """Evaluate a Python expression and return the result (repr).

    DISABLED by default. Set BATCH_AGENT_ENABLE_PYTHON_EVAL=1 to enable.
    """
    if os.getenv("BATCH_AGENT_ENABLE_PYTHON_EVAL") != "1":
        raise RuntimeError("python_eval is disabled unless BATCH_AGENT_ENABLE_PYTHON_EVAL=1")
    allowed_builtins = {"abs": abs, "len": len, "max": max, "min": min, "sum": sum, "range": range}
    return repr(eval(code, {"__builtins__": allowed_builtins}, {}))


@Tool.define(max_tokens=4000, cacheable=False, rate_limit=10)
async def claude_code(task: str, working_dir: str = ".") -> str:
    """Run a Claude Agent SDK subagent on a task.

    Requires: claude CLI installed + ANTHROPIC_API_KEY or Claude Max subscription.
    Falls back to error message if claude CLI is not available.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            task,
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ToolError(
            "claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        ) from exc

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise ToolError(f"claude CLI failed: {stderr.decode(errors='replace')[:500]}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.decode(errors="replace")
    return result.get("result", stdout.decode(errors="replace"))
