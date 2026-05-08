from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from . import BatchAgent


def main() -> None:
    parser = argparse.ArgumentParser(prog="batch-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(_run(args.spec))


async def _run(spec_path: str) -> None:
    spec = _load_spec(Path(spec_path))
    results = await BatchAgent.run(**spec)
    for result in results:
        print(json.dumps(_result_to_json(result), default=str, ensure_ascii=True))


def _load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("YAML specs require installing batch-agent[cli]") from exc
    return yaml.safe_load(text)


def _result_to_json(result: Any) -> dict[str, Any]:
    output = result.output
    if hasattr(output, "model_dump"):
        output = output.model_dump()
    elif hasattr(output, "dict"):
        output = output.dict()
    return {
        "job_id": result.job_id,
        "index": result.index,
        "ok": result.ok,
        "output": output,
        "error": None if result.error is None else result.error.__dict__,
        "attempts": result.attempts,
    }


if __name__ == "__main__":
    main()
