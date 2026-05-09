from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from . import BatchAgent
from .metrics import start_metrics_server
from .utils import to_jsonable


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="batch-agent",
        description="Batch Agent SDK — run task templates in parallel against LLM backends",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a batch spec file")
    run_parser.add_argument("--spec", required=True, help="Path to JSON or YAML spec file")
    run_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for SQLite checkpoint (enables crash recovery)",
    )
    run_parser.add_argument(
        "--no-hoist",
        action="store_true",
        default=False,
        help="Disable auto-hoisting of constant template variables into system prompt",
    )
    run_parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on this port (e.g. 9090)",
    )
    run_parser.add_argument(
        "--output",
        default="-",
        help="Output file path for JSON results (default: stdout)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        asyncio.run(_run(args))


async def _run(args: Any) -> None:
    spec_kwargs = _load_spec(Path(args.spec))

    # CLI flags override spec file values
    if args.checkpoint_dir:
        spec_kwargs["checkpoint_dir"] = args.checkpoint_dir
    if args.no_hoist:
        spec_kwargs["no_hoist"] = True

    # Start metrics server if requested
    metrics_server = None
    if args.metrics_port:
        from .scheduler import WaveScheduler
        from .compiler import TaskCompiler
        from .backends import backend_from_url
        from .spec import BatchSpec
        _spec = BatchSpec(**spec_kwargs)
        plan = TaskCompiler().compile(_spec)
        scheduler = WaveScheduler(plan, backend_from_url(_spec.backend))
        metrics_server = start_metrics_server(scheduler.metrics, port=args.metrics_port)
        results = await scheduler.run()
    else:
        results = await BatchAgent.run(**spec_kwargs)

    output_lines = [
        json.dumps(_result_to_json(r), default=str, ensure_ascii=True) for r in results
    ]
    output_text = "\n".join(output_lines) + "\n"

    if args.output == "-":
        sys.stdout.write(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8")

    if metrics_server:
        metrics_server.shutdown()


def _load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML specs require: pip install 'batch-agent[cli]'"
        ) from exc
    return yaml.safe_load(text)


def _result_to_json(result: Any) -> dict[str, Any]:
    return {
        "job_id": result.job_id,
        "index": result.index,
        "ok": result.ok,
        "output": to_jsonable(result.output),
        "error": None if result.error is None else result.error.__dict__,
        "attempts": result.attempts,
    }


if __name__ == "__main__":
    main()
