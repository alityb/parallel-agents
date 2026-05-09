"""Tests for --dashboard flag in cli.py.

The dashboard test runs a genuine end-to-end stream through the scheduler
with a mock backend and verifies the Rich layout renders without error.
We cannot test the terminal output visually but we can verify:
 - The dashboard code path executes to completion
 - All N results are written to the output file
 - No exceptions escape
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from batch_agent.backends import BackendAdapter, BackendResponse
from batch_agent.spec import AgentJob, Message, SharedContext


class FastMockBackend(BackendAdapter):
    async def generate(
        self, *, shared: SharedContext, job: AgentJob, messages: list[Message] | None = None,
        model: str, tools: list[dict] | None = None, metadata: dict | None = None,
        timeout: float | None = None,
    ) -> BackendResponse:
        body = json.dumps({"value": job.index})
        return BackendResponse(
            content=body,
            raw={"content": [{"type": "text", "text": body}]},
            stop_reason="end_turn",
        )


def _make_spec_file(n: int = 20) -> Path:
    spec = {
        "task": "Return JSON with value={idx}",
        "inputs": [{"idx": i} for i in range(n)],
        "model": "mock",
        "backend": "mock://",
        "max_concurrent": 5,
        "max_turns": 1,
        "max_retries": 0,
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(spec, f)
    f.close()
    return Path(f.name)


def test_dashboard_completes_without_error(tmp_path: Path) -> None:
    """Dashboard run with N=20 mock agents completes, all results written."""
    output_file = tmp_path / "results.jsonl"
    spec_path = _make_spec_file(20)

    async def run() -> None:
        # Import here to get the patched version
        from batch_agent.cli import _run_with_dashboard

        class FakeArgs:
            output = str(output_file)

        with patch("batch_agent.backends.backend_from_url", return_value=FastMockBackend()):
            spec_kwargs = json.loads(spec_path.read_text())
            await _run_with_dashboard(spec_kwargs, FakeArgs())

    asyncio.run(asyncio.wait_for(run(), timeout=30.0))

    # Verify all 20 results were written
    lines = [l for l in output_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 20, f"Expected 20 results, got {len(lines)}"
    for line in lines:
        obj = json.loads(line)
        assert obj["ok"] is True
        assert "value" in obj["output"]
    spec_path.unlink(missing_ok=True)


def test_dashboard_flag_in_help() -> None:
    """--dashboard flag must appear in `batch-agent run --help`."""
    import io
    from batch_agent.cli import main

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        main(["run", "--help"])
    except SystemExit:
        pass
    finally:
        sys.stdout = old

    assert "--dashboard" in buf.getvalue(), "--dashboard missing from help output"


def test_plain_run_unaffected_by_dashboard_flag(tmp_path: Path) -> None:
    """Without --dashboard, the normal code path runs unchanged."""
    spec_path = _make_spec_file(5)
    output_file = tmp_path / "out.jsonl"

    with patch("batch_agent.backends.backend_from_url", return_value=FastMockBackend()), \
         patch("batch_agent.BatchAgent._scheduler",
               staticmethod(lambda spec, backend=None: __import__("batch_agent.scheduler", fromlist=["WaveScheduler"]).WaveScheduler(
                   __import__("batch_agent.compiler", fromlist=["TaskCompiler"]).TaskCompiler().compile(spec),
                   FastMockBackend(),
               ))):
        from batch_agent.cli import main
        main(["run", "--spec", str(spec_path), "--output", str(output_file)])

    lines = [l for l in output_file.read_text().splitlines() if l.strip()]
    assert len(lines) == 5
    spec_path.unlink(missing_ok=True)
