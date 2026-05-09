"""Shared utilities for benchmark scripts and verification scripts.

All benchmark/stress-test scripts should import from here rather than
defining their own PASS/FAIL constants, report() helpers, or write_results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────

# Insert the repo root once so every benchmark can `import batch_agent`.
_REPO_ROOT = str(Path(__file__).parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── Verdict tags ──────────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def report(label: str, ok: bool, detail: str = "") -> bool:
    """Print a single PASS/FAIL verdict line and return ok."""
    tag = PASS if ok else FAIL
    print(f"{tag} {label}" + (f"  ({detail})" if detail else ""))
    return ok


# ── Tolerance helpers ─────────────────────────────────────────────────────────

def close(a: float, b: float, pct: float = 0.10) -> bool:
    """True if *a* is within ±pct of *b*."""
    return abs(a - b) <= b * pct


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def parser(name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(name)
    p.add_argument("--live", action="store_true",
                   help="Run against real backends when configured")
    p.add_argument("--output",
                   default=f"tests/benchmarks/results/{name}/results.json")
    return p


def write_results(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def base_result(name: str, live: bool, started: float) -> dict[str, Any]:
    return {
        "benchmark": name,
        "mode": "live" if live else "mock",
        "wall_clock_seconds": round(time.monotonic() - started, 4),
        "timestamp": time.time(),
    }


# ── pytest runner ─────────────────────────────────────────────────────────────

def run_pytest(
    test_path: str,
    *,
    cwd: str | None = None,
    extra_args: list[str] | None = None,
) -> tuple[bool, str]:
    """Run pytest on *test_path*; return (passed: bool, summary_line: str)."""
    cmd = ["python3", "-m", "pytest", test_path, "-q", "--tb=short",
           *(extra_args or [])]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=cwd or _REPO_ROOT,
    )
    last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "no output"
    return result.returncode == 0, last
