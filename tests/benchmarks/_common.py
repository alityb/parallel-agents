from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def parser(name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(name)
    p.add_argument("--live", action="store_true", help="Run against real backends when configured")
    p.add_argument("--output", default=f"tests/benchmarks/results/{name}/results.json")
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
