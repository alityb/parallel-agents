from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parent / "generated" / "code_review"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "opencode_baseline" / "results.json"
DEFAULT_BATCHCODE_PATH = ROOT.parent / "opencode" / "python"

sys.path.insert(0, str(ROOT))
if DEFAULT_BATCHCODE_PATH.exists():
    sys.path.insert(0, str(DEFAULT_BATCHCODE_PATH))


@dataclass(slots=True)
class ExpectedBug:
    bug_id: str
    line: int
    kind: str
    severity: str
    keywords: list[str]
    description: str


@dataclass(slots=True)
class ReviewTask:
    task_id: str
    path: str
    bugs: list[ExpectedBug]


@dataclass(slots=True)
class TaskResult:
    task_id: str
    path: str
    ok: bool
    wall_clock_seconds: float
    tool_calls: int
    found_bug_ids: list[str]
    expected_bug_count: int
    found_bug_count: int
    success: bool
    output_quality: int
    output_text: str
    prompt_sha256: str = ""
    output_sha256: str = ""
    normalized_output_sha256: str = ""
    error: str = ""
    # Token usage — populated from API usage fields when available
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0


def bug_functions() -> dict[str, list[str]]:
    return {
        "off_by_one": [
            "def moving_average(values: list[int], window: int) -> list[float]:",
            "    if window <= 0:",
            "        raise ValueError('window must be positive')",
            "    averages: list[float] = []",
            "    for start in range(0, len(values) - window):",
            "        chunk = values[start:start + window]",
            "        averages.append(sum(chunk) / window)",
            "    return averages",
        ],
        "missing_null_check": [
            "def customer_label(customer: dict[str, str] | None) -> str:",
            "    name = customer.get('name')",
            "    region = customer.get('region', 'unknown')",
            "    return f\"{name.strip().title()} ({region.upper()})\"",
        ],
        "wrong_variable": [
            "def invoice_total(lines: list[dict[str, float]]) -> float:",
            "    subtotal = 0.0",
            "    for line in lines:",
            "        subtotal += line['quantity'] * line['price']",
            "    discount = subtotal * 0.1 if subtotal > 1000 else 0.0",
            "    return subtotal - discunt",
        ],
    }


def build_tasks(fixture_dir: Path, n: int, seed: int) -> list[ReviewTask]:
    rng = random.Random(seed)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[ReviewTask] = []
    for index in range(n):
        lines: list[str] = []
        bugs: list[ExpectedBug] = []

        def add(value: str = "") -> int:
            lines.append(value)
            return len(lines)

        add('"""Synthetic code-review benchmark fixture."""')
        add("from __future__ import annotations")
        add("")
        add("")
        add("def clamp(value: int, lower: int, upper: int) -> int:")
        add("    if value < lower:")
        add("        return lower")
        add("    if value > upper:")
        add("        return upper")
        add("    return value")
        add("")
        add("")
        add("def normalize_slug(value: str) -> str:")
        add("    return '-'.join(part for part in value.lower().split() if part)")
        add("")
        add("")

        selected = rng.sample(list(bug_functions()), 2 + (index % 2))
        for kind in selected:
            add("")
            start = len(lines) + 1
            for source_line in bug_functions()[kind]:
                add(source_line)
            if kind == "off_by_one":
                bug_line = start + 4
                bugs.append(
                    ExpectedBug(
                        bug_id=f"{index}-off-by-one",
                        line=bug_line,
                        kind=kind,
                        severity="P1",
                        keywords=["off", "range", "missing", "last", "window", "boundary"],
                        description="Loop stops before the final valid window.",
                    )
                )
            if kind == "missing_null_check":
                bug_line = start + 1
                bugs.append(
                    ExpectedBug(
                        bug_id=f"{index}-missing-null-check",
                        line=bug_line,
                        kind=kind,
                        severity="P1",
                        keywords=["none", "null", "check", "attribute", "customer", "get"],
                        description="customer may be None before customer.get is called.",
                    )
                )
            if kind == "wrong_variable":
                bug_line = start + 5
                bugs.append(
                    ExpectedBug(
                        bug_id=f"{index}-wrong-variable",
                        line=bug_line,
                        kind=kind,
                        severity="P0",
                        keywords=["discunt", "discount", "undefined", "name", "variable", "typo"],
                        description="Return references undefined variable discunt instead of discount.",
                    )
                )

        add("")
        add("")
        add("def summarize_orders(orders: list[dict[str, int]]) -> dict[str, int]:")
        add("    total = 0")
        add("    count = 0")
        add("    for order in orders:")
        add("        total += order.get('amount', 0)")
        add("        count += 1")
        add("    return {'total': total, 'count': count}")
        add("")
        add("")
        add("def bucketize(values: list[int]) -> dict[str, int]:")
        add("    buckets = {'low': 0, 'mid': 0, 'high': 0}")
        add("    for value in values:")
        add("        if value < 10:")
        add("            buckets['low'] += 1")
        add("        elif value < 100:")
        add("            buckets['mid'] += 1")
        add("        else:")
        add("            buckets['high'] += 1")
        add("    return buckets")

        while len(lines) < 55 + (index % 16):
            add(f"# filler line {len(lines) + 1}: deterministic benchmark context")

        path = fixture_dir / f"synthetic_review_{index:02d}.py"
        path.write_text("\n".join(lines) + "\n")
        tasks.append(ReviewTask(task_id=f"review-{index:02d}", path=str(path), bugs=bugs))

    (fixture_dir / "expected_bugs.json").write_text(
        json.dumps([{"task_id": task.task_id, "path": task.path, "bugs": [asdict(bug) for bug in task.bugs]} for task in tasks], indent=2)
        + "\n"
    )
    return tasks


def load_tasks(fixture_dir: Path, n: int, seed: int, regenerate: bool) -> list[ReviewTask]:
    expected = fixture_dir / "expected_bugs.json"
    if regenerate or not expected.exists():
        return build_tasks(fixture_dir, n, seed)
    rows = json.loads(expected.read_text())
    return [
        ReviewTask(
            task_id=row["task_id"],
            path=row["path"],
            bugs=[ExpectedBug(**bug) for bug in row["bugs"]],
        )
        for row in rows[:n]
    ]


def prompt_for(path: str) -> str:
    return (
        "Review this Python file for correctness bugs. List each bug with: "
        "line number, description, severity (P0/P1/P2). "
        "Focus on real runtime or logic bugs, not style. Be concise.\n\n"
        f"File contents:\n\n{Path(path).read_text()}"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_output_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def result_fingerprints(prompt: str, output: str) -> dict[str, str]:
    return {
        "prompt_sha256": sha256_text(prompt),
        "output_sha256": sha256_text(output),
        "normalized_output_sha256": sha256_text(normalize_output_text(output)),
    }


async def discover_openai_model(base_url: str, api_key: str) -> str:
    preferred = os.environ.get("OPENAI_MODEL")
    if preferred:
        return preferred
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/models",
            headers={"authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
    ids = {item.get("id") for item in response.json().get("data", []) if isinstance(item, dict)}
    for candidate in ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o"]:
        if candidate in ids:
            return candidate
    raise RuntimeError("No supported OpenAI chat model found. Set OPENAI_MODEL explicitly.")


async def run_openai_task(task: ReviewTask, base_url: str, api_key: str, model: str, timeout: float, max_tokens: int) -> TaskResult:
    started = time.monotonic()
    prompt = prompt_for(task.path)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
        wall = time.monotonic() - started
        payload = response.json()
        choice = (payload.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        text = message.get("content") or json.dumps(payload)
        found, quality = score_output(text, task.bugs)
        fingerprints = result_fingerprints(prompt, text)
        usage = payload.get("usage") or {}
        cached = (
            usage.get("cached_tokens")
            or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            or 0
        )
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=True,
            wall_clock_seconds=wall,
            tool_calls=count_tool_calls(payload),
            found_bug_ids=found,
            expected_bug_count=len(task.bugs),
            found_bug_count=len(found),
            success=len(found) == len(task.bugs),
            output_quality=quality,
            output_text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            cached_tokens=cached,
            completion_tokens=usage.get("completion_tokens", 0),
            **fingerprints,
        )
    except Exception as error:
        wall = time.monotonic() - started
        fingerprints = result_fingerprints(prompt, "")
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=False,
            wall_clock_seconds=wall,
            tool_calls=0,
            found_bug_ids=[],
            expected_bug_count=len(task.bugs),
            found_bug_count=0,
            success=False,
            output_quality=1,
            output_text="",
            **fingerprints,
            error=repr(error),
        )


async def run_openai_sequential(tasks: list[ReviewTask], base_url: str, model: str, timeout: float, max_tokens: int) -> tuple[list[TaskResult], float, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI baseline")
    selected_model = model or await discover_openai_model(base_url, api_key)
    results: list[TaskResult] = []
    started = time.monotonic()
    for task in tasks:
        results.append(await run_openai_task(task, base_url, api_key, selected_model, timeout, max_tokens))
    return results, time.monotonic() - started, selected_model


async def run_sglang_task(task: ReviewTask, base_url: str, model: str, timeout: float, max_tokens: int) -> TaskResult:
    started = time.monotonic()
    prompt = prompt_for(task.path)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                headers={"content-type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
        wall = time.monotonic() - started
        payload = response.json()
        choice = (payload.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        text = message.get("content") or json.dumps(payload)
        found, quality = score_output(text, task.bugs)
        fingerprints = result_fingerprints(prompt, text)
        usage = payload.get("usage") or {}
        cached = (
            usage.get("cached_tokens")
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            or 0
        )
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=True,
            wall_clock_seconds=wall,
            tool_calls=count_tool_calls(payload),
            found_bug_ids=found,
            expected_bug_count=len(task.bugs),
            found_bug_count=len(found),
            success=len(found) == len(task.bugs),
            output_quality=quality,
            output_text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            cached_tokens=cached,
            completion_tokens=usage.get("completion_tokens", 0),
            **fingerprints,
        )
    except Exception as error:
        wall = time.monotonic() - started
        fingerprints = result_fingerprints(prompt, "")
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=False,
            wall_clock_seconds=wall,
            tool_calls=0,
            found_bug_ids=[],
            expected_bug_count=len(task.bugs),
            found_bug_count=0,
            success=False,
            output_quality=1,
            output_text="",
            **fingerprints,
            error=repr(error),
        )


async def run_sglang_sequential(tasks: list[ReviewTask], backend_url: str, model: str, timeout: float, max_tokens: int) -> tuple[list[TaskResult], float]:
    base_url = backend_url.replace("sglang://", "http://", 1) if backend_url.startswith("sglang://") else backend_url
    results: list[TaskResult] = []
    started = time.monotonic()
    for task in tasks:
        results.append(await run_sglang_task(task, base_url, model, timeout, max_tokens))
    return results, time.monotonic() - started


def text_from_claude_json(stdout: str) -> tuple[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, None
    if isinstance(payload, dict):
        for key in ["result", "content", "message"]:
            if isinstance(payload.get(key), str):
                return payload[key], payload
        return json.dumps(payload, indent=2), payload
    return str(payload), payload


def count_tool_calls(value: Any) -> int:
    if isinstance(value, dict):
        own = 1 if value.get("type") in {"tool_use", "tool_result"} else 0
        nested = sum(count_tool_calls(child) for child in value.values())
        calls = value.get("tool_calls")
        if isinstance(calls, list):
            nested += len(calls)
        return own + nested
    if isinstance(value, list):
        return sum(count_tool_calls(item) for item in value)
    return 0


def score_output(text: str, bugs: list[ExpectedBug]) -> tuple[list[str], int]:
    lower = text.lower()
    found: list[str] = []
    for bug in bugs:
        line_hit = any(
            re.search(pattern, lower)
            for delta in (-1, 0, 1)
            for pattern in [
                rf"\bline\s*{bug.line + delta}\b",
                rf"\bl{bug.line + delta}\b",
                rf"(?<!\d){bug.line + delta}(?!\d)",
            ]
        )
        keyword_hit = any(keyword.lower() in lower for keyword in bug.keywords)
        if line_hit and keyword_hit:
            found.append(bug.bug_id)

    coverage = len(found) / max(1, len(bugs))
    if coverage == 1:
        quality = 5
    elif coverage >= 0.67:
        quality = 4
    elif coverage >= 0.34:
        quality = 3
    elif found:
        quality = 2
    else:
        quality = 1
    if found and not re.search(r"\bP[012]\b", text):
        quality = max(1, quality - 1)
    return found, quality


def run_claude_task(task: ReviewTask, timeout: float) -> TaskResult:
    started = time.monotonic()
    prompt = prompt_for(task.path)
    proc = subprocess.run(
        [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    wall = time.monotonic() - started
    text, payload = text_from_claude_json(proc.stdout)
    error = proc.stderr.strip() if proc.returncode else ""
    if proc.returncode:
        text = text or proc.stdout or error
    found, quality = score_output(text, task.bugs)
    fingerprints = result_fingerprints(prompt, text)
    return TaskResult(
        task_id=task.task_id,
        path=task.path,
        ok=proc.returncode == 0,
        wall_clock_seconds=wall,
        tool_calls=count_tool_calls(payload),
        found_bug_ids=found,
        expected_bug_count=len(task.bugs),
        found_bug_count=len(found),
        success=len(found) == len(task.bugs),
        output_quality=quality if proc.returncode == 0 else 1,
        output_text=text,
        **fingerprints,
        error=error,
    )


async def run_batchcode_task(task: ReviewTask, backend: Any, model: str, working_root: Path) -> TaskResult:
    """Run one task through the BatchAgent backend directly.

    When the optional ``batchcode`` package is installed this uses the full
    OpenCode agent loop.  Otherwise it falls back to a single generate() call
    through the same backend, which is sufficient for measuring token usage.
    """
    started = time.monotonic()
    prompt = prompt_for(task.path)

    try:
        from batchcode.agent import OpenCodeAgent  # type: ignore[import]
        agent = OpenCodeAgent(
            job_id=task.task_id,
            working_dir=str(working_root / task.task_id),
            discovery_bus=None,
            agent_pool=None,
        )
        result = await agent.run_turn([{"role": "user", "content": prompt}], backend, [])
        wall = time.monotonic() - started
        found, quality = score_output(result.text, task.bugs)
        fingerprints = result_fingerprints(prompt, result.text)
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=True,
            wall_clock_seconds=wall,
            tool_calls=len(result.tool_calls),
            found_bug_ids=found,
            expected_bug_count=len(task.bugs),
            found_bug_count=len(found),
            success=len(found) == len(task.bugs),
            output_quality=quality,
            output_text=result.text,
            **fingerprints,
        )
    except ModuleNotFoundError:
        pass  # fall through to direct generate() below
    except Exception as error:
        wall = time.monotonic() - started
        fingerprints = result_fingerprints(prompt, "")
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=False,
            wall_clock_seconds=wall,
            tool_calls=0,
            found_bug_ids=[],
            expected_bug_count=len(task.bugs),
            found_bug_count=0,
            success=False,
            output_quality=1,
            output_text="",
            **fingerprints,
            error=repr(error),
        )

    # Fallback: single generate() call through the backend.
    # Token counts are still accumulated by TokenTrackingSGLangBackend.
    try:
        from batch_agent.spec import AgentJob, SharedContext, Message
        shared = SharedContext(prefix="", strip_preamble=False)
        job = AgentJob(
            job_id=task.task_id,
            index=0,
            input_data={},
            prompt=prompt,
            estimated_prompt_tokens=len(prompt) // 4,
        )
        resp = await backend.generate(
            shared=shared,
            job=job,
            model=model,
            messages=[Message(role="user", content=prompt)],
        )
        wall = time.monotonic() - started
        text = resp.content or ""
        found, quality = score_output(text, task.bugs)
        fingerprints = result_fingerprints(prompt, text)
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=True,
            wall_clock_seconds=wall,
            tool_calls=0,
            found_bug_ids=found,
            expected_bug_count=len(task.bugs),
            found_bug_count=len(found),
            success=len(found) == len(task.bugs),
            output_quality=quality,
            output_text=text,
            **fingerprints,
        )
    except Exception as error:
        import traceback as _tb
        _tb.print_exc()
        wall = time.monotonic() - started
        fingerprints = result_fingerprints(prompt, "")
        return TaskResult(
            task_id=task.task_id,
            path=task.path,
            ok=False,
            wall_clock_seconds=wall,
            tool_calls=0,
            found_bug_ids=[],
            expected_bug_count=len(task.bugs),
            found_bug_count=0,
            success=False,
            output_quality=1,
            output_text="",
            **fingerprints,
            error=repr(error),
        )


async def run_batchcode(tasks: list[ReviewTask], backend_url: str, model: str, max_inflight: int, working_root: Path, max_tokens: int) -> tuple[list[TaskResult], float, dict[str, int]]:
    from batch_agent.backends.sglang import SGLangBackend

    class TokenTrackingSGLangBackend(SGLangBackend):
        """SGLang backend that accumulates token usage across all generate() calls."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._total_prompt_tokens: int = 0
            self._total_cached_tokens: int = 0
            self._total_completion_tokens: int = 0
            self._lock = asyncio.Lock()

        @property
        def usage_totals(self) -> dict[str, int]:
            return {
                "prompt_tokens": self._total_prompt_tokens,
                "cached_tokens": self._total_cached_tokens,
                "completion_tokens": self._total_completion_tokens,
                "actual_prefill_tokens": self._total_prompt_tokens - self._total_cached_tokens,
            }

        async def generate(
            self,
            *,
            shared: Any,
            job: Any,
            messages: list[Any] | None = None,
            model: str,
            tools: list[dict[str, Any]] | None = None,
            metadata: dict[str, Any] | None = None,
            timeout: float | None = None,
        ) -> Any:
            metadata = dict(metadata or {})
            extensions = dict(metadata.get("request_extensions") or {})
            extensions.setdefault("max_tokens", max_tokens)
            extensions.setdefault("temperature", 0)
            metadata["request_extensions"] = extensions
            resp = await super().generate(
                shared=shared,
                job=job,
                messages=messages,
                model=model,
                tools=tools,
                metadata=metadata,
                timeout=timeout,
            )
            # Accumulate token usage from the raw API response
            if isinstance(resp.raw, dict):
                usage = resp.raw.get("usage") or {}
                cached = (
                    usage.get("cached_tokens")
                    or usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    or 0
                )
                async with self._lock:
                    self._total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self._total_cached_tokens += cached
                    self._total_completion_tokens += usage.get("completion_tokens", 0)
            return resp

    os.environ["BATCHCODE_MODEL"] = model
    backend = TokenTrackingSGLangBackend.from_url(backend_url)
    semaphore = asyncio.Semaphore(max_inflight)
    started = time.monotonic()

    async def one(task: ReviewTask) -> TaskResult:
        async with semaphore:
            return await run_batchcode_task(task, backend, model, working_root)

    results = await asyncio.gather(*[one(task) for task in tasks])
    return list(results), time.monotonic() - started, backend.usage_totals


def aggregate(name: str, results: list[TaskResult], wall_clock_seconds: float, extra_usage: dict[str, int] | None = None) -> dict[str, Any]:
    expected = sum(result.expected_bug_count for result in results)
    found = sum(result.found_bug_count for result in results)
    ok = sum(1 for result in results if result.ok)
    successful = sum(1 for result in results if result.success)

    # Token totals: use per-task fields when populated (sequential path),
    # fall back to extra_usage from backend accumulator (BatchAgent path).
    per_task_prompt = sum(result.prompt_tokens for result in results)
    per_task_cached = sum(result.cached_tokens for result in results)
    per_task_completion = sum(result.completion_tokens for result in results)
    if extra_usage and per_task_prompt == 0:
        total_prompt = extra_usage.get("prompt_tokens", 0)
        total_cached = extra_usage.get("cached_tokens", 0)
        total_completion = extra_usage.get("completion_tokens", 0)
    else:
        total_prompt = per_task_prompt
        total_cached = per_task_cached
        total_completion = per_task_completion
    actual_prefill = total_prompt - total_cached

    n = max(1, len(results))
    return {
        "name": name,
        "ok": ok,
        "failed": len(results) - ok,
        "tasks": len(results),
        "wall_clock_seconds": wall_clock_seconds,
        "success_rate": successful / n,
        "bug_recall": found / max(1, expected),
        "expected_bugs": expected,
        "found_bugs": found,
        "tool_calls": sum(result.tool_calls for result in results),
        "quality_avg": sum(result.output_quality for result in results) / n,
        "per_task_wall_clock_seconds": [result.wall_clock_seconds for result in results],
        # Token tracking
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "total_completion_tokens": total_completion,
        "actual_prefill_tokens": actual_prefill,
        "per_agent_prompt_tokens": total_prompt / n,
        "per_agent_actual_prefill": actual_prefill / n,
    }


def result_rows(results: list[TaskResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def output_equivalence_audit(
    reference_name: str,
    reference_results: list[TaskResult],
    candidate_name: str,
    candidate_results: list[TaskResult],
) -> dict[str, Any]:
    """Compare final outputs for task-level equivalence.

    Exact text equality is too strict for LLM generations. For the code-review
    benchmark, the operational equivalence criterion is the set of expected bug
    IDs found for each task. We still record exact/normalized output hash match
    rates so claims can distinguish task-equivalent from byte-identical output.
    """

    reference_by_id = {result.task_id: result for result in reference_results}
    candidate_by_id = {result.task_id: result for result in candidate_results}
    task_ids = sorted(set(reference_by_id) & set(candidate_by_id))
    rows: list[dict[str, Any]] = []

    for task_id in task_ids:
        ref = reference_by_id[task_id]
        cand = candidate_by_id[task_id]
        ref_ids = sorted(ref.found_bug_ids)
        cand_ids = sorted(cand.found_bug_ids)
        rows.append(
            {
                "task_id": task_id,
                "prompt_hash_match": bool(ref.prompt_sha256 and ref.prompt_sha256 == cand.prompt_sha256),
                "finding_set_match": ref_ids == cand_ids,
                "success_match": ref.success == cand.success,
                "exact_output_match": bool(ref.output_sha256 and ref.output_sha256 == cand.output_sha256),
                "normalized_output_match": bool(
                    ref.normalized_output_sha256
                    and ref.normalized_output_sha256 == cand.normalized_output_sha256
                ),
                "reference_found_bug_ids": ref_ids,
                "candidate_found_bug_ids": cand_ids,
                "reference_quality": ref.output_quality,
                "candidate_quality": cand.output_quality,
            }
        )

    tasks_compared = len(rows)

    def rate(key: str) -> float:
        return sum(1 for row in rows if row[key]) / max(1, tasks_compared)

    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "tasks_compared": tasks_compared,
        "reference_only_task_ids": sorted(set(reference_by_id) - set(candidate_by_id)),
        "candidate_only_task_ids": sorted(set(candidate_by_id) - set(reference_by_id)),
        "prompt_hash_match_rate": rate("prompt_hash_match"),
        "finding_set_match_rate": rate("finding_set_match"),
        "success_match_rate": rate("success_match"),
        "exact_output_match_rate": rate("exact_output_match"),
        "normalized_output_match_rate": rate("normalized_output_match"),
        "task_equivalent": tasks_compared > 0 and rate("finding_set_match") == 1.0,
        "byte_identical": tasks_compared > 0 and rate("exact_output_match") == 1.0,
        "rows": rows,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["both", "openai", "sglang", "sglang-vs-batchcode", "claude", "batchcode"], default="both")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--claude-timeout", type=float, default=180)
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-model", default=os.environ.get("OPENAI_MODEL", ""))
    parser.add_argument("--openai-timeout", type=float, default=120)
    parser.add_argument("--backend", default="sglang://localhost:30000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--max-inflight", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--sglang-timeout", type=float, default=120)
    args = parser.parse_args()

    tasks = load_tasks(args.fixture_dir, args.n, args.seed, args.regenerate)
    payload: dict[str, Any] = {
        "benchmark": "opencode_baseline_code_review",
        "n": len(tasks),
        "seed": args.seed,
        "fixture_dir": str(args.fixture_dir),
        "model": args.model,
        "backend": args.backend,
        "openai_model": args.openai_model or os.environ.get("OPENAI_MODEL", ""),
        "openai_base_url": args.openai_base_url,
        "max_inflight": args.max_inflight,
        "started_at": time.time(),
    }

    if args.mode in {"both", "openai"}:
        openai_results, openai_wall, selected_model = await run_openai_sequential(
            tasks,
            args.openai_base_url,
            args.openai_model,
            args.openai_timeout,
            args.max_tokens,
        )
        payload["openai_model"] = selected_model
        payload["openai_sequential"] = {
            "summary": aggregate("openai_sequential", openai_results, openai_wall),
            "results": result_rows(openai_results),
        }

    if args.mode in {"sglang", "sglang-vs-batchcode"}:
        sglang_results, sglang_wall = await run_sglang_sequential(
            tasks,
            args.backend,
            args.model,
            args.sglang_timeout,
            args.max_tokens,
        )
        payload["sglang_sequential"] = {
            "summary": aggregate("sglang_sequential", sglang_results, sglang_wall),
            "results": result_rows(sglang_results),
        }

    if args.mode == "claude":
        started = time.monotonic()
        claude_results = [run_claude_task(task, args.claude_timeout) for task in tasks]
        payload["claude_sequential"] = {
            "summary": aggregate("claude_sequential", claude_results, time.monotonic() - started),
            "results": result_rows(claude_results),
        }

    if args.mode in {"both", "batchcode", "sglang-vs-batchcode"}:
        batch_results, batch_wall, batch_usage = await run_batchcode(
            tasks,
            args.backend,
            args.model,
            args.max_inflight,
            args.fixture_dir / "working" / "batchcode",
            args.max_tokens,
        )
        payload["batchcode_parallel"] = {
            "summary": aggregate("batchcode_parallel", batch_results, batch_wall, extra_usage=batch_usage),
            "results": result_rows(batch_results),
        }

    if "openai_sequential" in payload and "batchcode_parallel" in payload:
        openai_wall = payload["openai_sequential"]["summary"]["wall_clock_seconds"]
        batch_wall = payload["batchcode_parallel"]["summary"]["wall_clock_seconds"]
        output_audit = output_equivalence_audit(
            "openai_sequential",
            openai_results,
            "batchcode_parallel",
            batch_results,
        )
        seq_prefill = payload["openai_sequential"]["summary"]["actual_prefill_tokens"]
        batch_prefill = payload["batchcode_parallel"]["summary"]["actual_prefill_tokens"]
        payload["comparison"] = {
            "speedup_openai_over_batchcode": openai_wall / batch_wall if batch_wall else None,
            "wall_clock_seconds_saved": openai_wall - batch_wall,
            "quality_avg_delta_batchcode_minus_openai": (
                payload["batchcode_parallel"]["summary"]["quality_avg"]
                - payload["openai_sequential"]["summary"]["quality_avg"]
            ),
            "bug_recall_delta_batchcode_minus_openai": (
                payload["batchcode_parallel"]["summary"]["bug_recall"]
                - payload["openai_sequential"]["summary"]["bug_recall"]
            ),
            "prefill_tokens_sequential": seq_prefill,
            "prefill_tokens_batchagent": batch_prefill,
            "prefill_reduction_pct": (
                round(100 * (seq_prefill - batch_prefill) / seq_prefill, 1)
                if seq_prefill else None
            ),
            "output_equivalence": output_audit,
        }

    if "sglang_sequential" in payload and "batchcode_parallel" in payload:
        sglang_wall = payload["sglang_sequential"]["summary"]["wall_clock_seconds"]
        batch_wall = payload["batchcode_parallel"]["summary"]["wall_clock_seconds"]
        output_audit = output_equivalence_audit(
            "sglang_sequential",
            sglang_results,
            "batchcode_parallel",
            batch_results,
        )
        seq_prefill = payload["sglang_sequential"]["summary"]["actual_prefill_tokens"]
        batch_prefill = payload["batchcode_parallel"]["summary"]["actual_prefill_tokens"]
        seq_prompt = payload["sglang_sequential"]["summary"]["total_prompt_tokens"]
        batch_prompt = payload["batchcode_parallel"]["summary"]["total_prompt_tokens"]
        payload["comparison"] = {
            "speedup_sglang_sequential_over_batchcode": sglang_wall / batch_wall if batch_wall else None,
            "wall_clock_seconds_saved": sglang_wall - batch_wall,
            "quality_avg_delta_batchcode_minus_sglang_sequential": (
                payload["batchcode_parallel"]["summary"]["quality_avg"]
                - payload["sglang_sequential"]["summary"]["quality_avg"]
            ),
            "bug_recall_delta_batchcode_minus_sglang_sequential": (
                payload["batchcode_parallel"]["summary"]["bug_recall"]
                - payload["sglang_sequential"]["summary"]["bug_recall"]
            ),
            "prefill_tokens_sequential": seq_prefill,
            "prefill_tokens_batchagent": batch_prefill,
            "prefill_reduction_pct": (
                round(100 * (seq_prefill - batch_prefill) / seq_prefill, 1)
                if seq_prefill else None
            ),
            "total_prompt_tokens_sequential": seq_prompt,
            "total_prompt_tokens_batchagent": batch_prompt,
            "output_equivalence": output_audit,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["summary"] for key, value in payload.items() if isinstance(value, dict) and "summary" in value}, indent=2, sort_keys=True))
    if "comparison" in payload:
        print(json.dumps({"comparison": payload["comparison"]}, indent=2, sort_keys=True))
    print(f"output={args.output}")


if __name__ == "__main__":
    asyncio.run(main())
