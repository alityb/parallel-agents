"""Shared utilities used across multiple batch_agent modules.

Centralising here prevents the same logic from being copy-pasted
across backends, compiler, __init__, checkpoint, cli, and distributed.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from typing import Any


# ── Pydantic / dict serialisation ─────────────────────────────────────────────

def to_jsonable(value: Any) -> Any:
    """Convert a Pydantic model instance (or any value) to a JSON-serialisable form.

    Handles both Pydantic v1 (.dict()) and v2 (.model_dump()).
    Falls through to the original value for anything else.
    """
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def extract_schema(output_schema: Any) -> dict[str, Any] | None:
    """Extract a JSON Schema dict from a Pydantic model class or a raw schema dict.

    Returns None when output_schema is None.
    Raises TypeError for unrecognised types.
    """
    if output_schema is None:
        return None
    if hasattr(output_schema, "model_json_schema"):
        return output_schema.model_json_schema()
    if hasattr(output_schema, "schema"):
        return output_schema.schema()
    if isinstance(output_schema, dict):
        return output_schema
    raise TypeError(
        f"output_schema must be a Pydantic model class or JSON schema dict, got {type(output_schema)}"
    )


# ── Percentile helpers ─────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float | None:
    """Return the p-th percentile (0–1) of *values*, or None if empty.

    Uses the nearest-rank method (ceil).
    """
    if not values:
        return None
    import math
    data = sorted(values)
    idx = math.ceil(len(data) * p) - 1
    return data[max(0, min(idx, len(data) - 1))]


def p50(values: list[float]) -> float | None:
    return percentile(values, 0.50)


def p75(values: list[float]) -> float | None:
    return percentile(values, 0.75)


def p95(values: list[float]) -> float | None:
    return percentile(values, 0.95)


def p99(values: list[float]) -> float | None:
    return percentile(values, 0.99)


# ── Prefix KV hashing ──────────────────────────────────────────────────────────

# Headers that vary per session and poison the prefix cache.
# Source: NVIDIA Dynamo blog, May 2026 — confirmed on Claude Code prompts.
_PREAMBLE_PATTERN = re.compile(
    r"^(x-anthropic-[^\n]*\n|x-amz-[^\n]*\n)",
    re.MULTILINE | re.IGNORECASE,
)


def strip_preamble_headers(prompt: str) -> str:
    """Remove session-variant headers before prefix hashing and cache warming.

    Claude Code prepends a session-specific billing header at token zero:
        x-anthropic-billing-header: cc_version=0.2.93; cch=abc123==
    This header changes per session, making every agent's prefix unique and
    defeating prefix caching entirely. Strip it before tokenization.

    Reference: NVIDIA Dynamo blog May 2026, --strip-anthropic-preamble flag.
    """
    return _PREAMBLE_PATTERN.sub("", prompt)


def prefix_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8.

    Used as a stable, opaque key for the shared system-prompt KV cache.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hex digest of *value* serialised as canonical JSON.

    Used by ToolPool to generate per-call cache keys.
    """
    import json as _json
    encoded = _json.dumps(value, sort_keys=True, default=str, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ── HTTP constants ─────────────────────────────────────────────────────────────

#: Timeout for internal probe / admin HTTP calls (metrics, prefetch, pin-blocks).
INTERNAL_HTTP_TIMEOUT: float = 5.0

#: Timeout for prefix warm-up HTTP calls to the inference backend.
PREFIX_WARM_TIMEOUT: float = 30.0

#: Timeout for tool HTTP calls (web_search, http_get) that lack an explicit timeout.
TOOL_HTTP_TIMEOUT: float = 15.0

#: Placeholder API key used by self-hosted backends that need a non-empty value.
NO_API_KEY: str = "EMPTY"

#: Default max_tokens budget passed to inference backends when not specified per-call.
DEFAULT_MAX_TOKENS: int = 4096


# ── Prometheus scraper ────────────────────────────────────────────────────────

import re as _re

_METRIC_LINE_RE = _re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:.]*)(?:\{[^}]*\})?\s+([\d.e+\-]+)")


def parse_prometheus_metrics(
    text: str,
    *,
    prefix: str = "",
    skip_suffixes: tuple[str, ...] = ("_created",),
) -> dict[str, float]:
    """Parse Prometheus text-format metrics, returning name → float.

    Args:
        text:            Raw Prometheus text output.
        prefix:          Optional metric-name prefix to filter (e.g. "vllm:").
        skip_suffixes:   Metric names ending with these are dropped (timestamp gauges).

    Only lines that match the Prometheus metric line format (possibly with
    label groups ``{…}``) are returned.  Comment lines (``#``) are skipped.
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, value_str = m.group(1), m.group(2)
        if prefix and not name.startswith(prefix):
            continue
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        try:
            result[name] = float(value_str)
        except ValueError:
            continue
    return result
