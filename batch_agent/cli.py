from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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
    run_parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Show live Rich dashboard: progress, results stream, metrics, timing",
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

    if args.dashboard:
        await _run_with_dashboard(spec_kwargs, args)
        return

    # ── plain run ──────────────────────────────────────────────────────────────
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


# ── Dashboard ──────────────────────────────────────────────────────────────────

async def _run_with_dashboard(spec_kwargs: dict[str, Any], args: Any) -> None:
    """Run with a live Rich terminal dashboard.

    Four panels:
      ┌─────────────────────────────────────────────────────┐
      │ Progress bar          │ Live results (last 5)       │
      ├───────────────────────┼─────────────────────────────┤
      │ Metrics               │ Timing                      │
      └───────────────────────┴─────────────────────────────┘

    Polls scheduler.metrics every 500 ms.  No Prometheus required.
    """
    try:
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
        from rich.table import Table
        from rich.text import Text
        from rich.console import Console
    except ImportError:
        print("Dashboard requires rich: pip install rich", file=sys.stderr)
        sys.exit(1)

    from .scheduler import WaveScheduler
    from .compiler import TaskCompiler
    from .backends import backend_from_url
    from .spec import BatchSpec

    spec = BatchSpec(**spec_kwargs)
    plan = TaskCompiler().compile(spec)
    backend = backend_from_url(spec.backend)
    scheduler = WaveScheduler(plan, backend)

    n_total = len(plan.jobs)
    results: list[Any] = []
    recent_results: list[str] = []    # last 5 lines shown in stream panel
    ok_count = 0
    fail_count = 0
    started_at = time.monotonic()

    # ── layout ─────────────────────────────────────────────────────────────────

    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=2),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].split_row(
        Layout(name="progress", ratio=1),
        Layout(name="stream",   ratio=2),
    )
    layout["bottom"].split_row(
        Layout(name="metrics", ratio=1),
        Layout(name="timing",  ratio=1),
    )

    def _render_progress(done: int, total: int, ok: int, fail: int) -> Panel:
        bar_width = 40
        filled = int(bar_width * done / max(total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = 100 * done / max(total, 1)
        lines = [
            f"[bold cyan]{bar}[/bold cyan] {pct:.0f}%",
            f"  [green]{ok} OK[/green]  [red]{fail} failed[/red]  {done}/{total} total",
        ]
        return Panel("\n".join(lines), title="[bold]Progress[/bold]", border_style="cyan")

    def _render_stream(lines: list[str]) -> Panel:
        content = "\n".join(lines[-5:]) if lines else "[dim]waiting for first result...[/dim]"
        return Panel(content, title="[bold]Live results[/bold]", border_style="green")

    def _render_metrics(sched: WaveScheduler) -> Panel:
        m = sched.metrics
        sem = sched._semaphore
        tput = ok_count / max(time.monotonic() - started_at, 0.001)
        lines = [
            f"  Cache hit rate  : [cyan]{getattr(m, 'cache_hit_rate', 0.0):.1%}[/cyan]",
            f"  In-flight       : [yellow]{sem.active}[/yellow] / {sem.capacity}",
            f"  Waiting (sem)   : [yellow]{sem.waiting}[/yellow]",
            f"  Throughput      : [green]{tput:.1f} agents/s[/green]",
        ]
        return Panel("\n".join(lines), title="[bold]Metrics[/bold]", border_style="yellow")

    def _render_timing(done: int, total: int) -> Panel:
        elapsed = time.monotonic() - started_at
        if done > 0 and done < total:
            eta = elapsed / done * (total - done)
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "—"
        lines = [
            f"  Elapsed : [cyan]{elapsed:.1f}s[/cyan]",
            f"  ETA     : [cyan]{eta_str}[/cyan]",
            f"  Started : [dim]{time.strftime('%H:%M:%S', time.localtime(time.time() - elapsed))}[/dim]",
        ]
        return Panel("\n".join(lines), title="[bold]Timing[/bold]", border_style="magenta")

    def _refresh() -> None:
        layout["progress"].update(_render_progress(len(results), n_total, ok_count, fail_count))
        layout["stream"].update(_render_stream(recent_results))
        layout["metrics"].update(_render_metrics(scheduler))
        layout["timing"].update(_render_timing(len(results), n_total))

    # ── run ────────────────────────────────────────────────────────────────────

    console = Console()
    with Live(layout, console=console, refresh_per_second=2, screen=False) as live:
        async def stream_results() -> None:
            nonlocal ok_count, fail_count
            async for result in scheduler.stream():
                results.append(result)
                if result.ok:
                    ok_count += 1
                    out_str = str(to_jsonable(result.output))[:60]
                    recent_results.append(
                        f"[green]✓[/green] {result.job_id}  {out_str}"
                    )
                else:
                    fail_count += 1
                    recent_results.append(
                        f"[red]✗[/red] {result.job_id}  {result.error.message[:60]}"
                    )
                _refresh()

        async def poll_loop() -> None:
            while len(results) < n_total:
                _refresh()
                await asyncio.sleep(0.5)

        _refresh()
        await asyncio.gather(stream_results(), poll_loop())

    # ── write output after dashboard closes ────────────────────────────────────

    output_lines = [
        json.dumps(_result_to_json(r), default=str, ensure_ascii=True)
        for r in results
    ]
    output_text = "\n".join(output_lines) + "\n"

    if args.output == "-":
        sys.stdout.write(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8")


# ── helpers ────────────────────────────────────────────────────────────────────

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
