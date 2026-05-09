"""Scheduler metrics — in-process counters/gauges written on every turn.

Exposes a /metrics endpoint in Prometheus text format.
Does NOT require the prometheus_client library.

Metrics recorded:
  batch_agent_turns_total              counter  (per job_id label)
  batch_agent_generate_latency_seconds histogram buckets
  batch_agent_jobs_completed_total     counter
  batch_agent_jobs_failed_total        counter
  batch_agent_concurrent_active        gauge
  batch_agent_cache_hit_rate           gauge     (updated by adaptive controller)
  batch_agent_concurrency_current      gauge
  batch_agent_concurrency_changes_total counter
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from typing import Any


@dataclass
class SchedulerMetrics:
    # ── counters ───────────────────────────────────────────────────────────────
    turns_total: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    concurrency_changes: int = 0

    # ── gauges ─────────────────────────────────────────────────────────────────
    concurrent_active: int = 0
    cache_hit_rate: float = 0.0
    concurrency_current: int = 0

    # ── histograms (generate latency buckets in seconds) ───────────────────────
    _latency_buckets: dict[float, int] = field(
        default_factory=lambda: {b: 0 for b in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, math.inf]}
    )
    _latency_sum: float = 0.0
    _latency_count: int = 0

    # ── per-job turn latency (last N) ──────────────────────────────────────────
    _job_turn_latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    # ── concurrency change log ─────────────────────────────────────────────────
    _concurrency_log: list[dict[str, Any]] = field(default_factory=list)

    # ── reset / read helpers ────────────────────────────────────────────────────
    started_at: float = field(default_factory=time.time)

    def record_turn(self, job_id: str, latency_seconds: float) -> None:
        self.turns_total += 1
        self._latency_sum += latency_seconds
        self._latency_count += 1
        for bucket in self._latency_buckets:
            if latency_seconds <= bucket:
                self._latency_buckets[bucket] += 1
        self._job_turn_latencies[job_id].append(latency_seconds)

    def record_completion(self) -> None:
        self.jobs_completed += 1

    def record_failure(self) -> None:
        self.jobs_failed += 1

    def record_cache_hit_rate(self, rate: float) -> None:
        self.cache_hit_rate = rate

    def record_concurrency_change(self, old: int, new: int, direction: str) -> None:
        self.concurrency_changes += 1
        self.concurrency_current = new
        self._concurrency_log.append({
            "ts": time.time(), "old": old, "new": new, "direction": direction,
        })

    def p50_latency(self) -> float | None:
        if not self._latency_count:
            return None
        target = self._latency_count * 0.50
        running = 0
        for bucket, count in self._latency_buckets.items():
            running += count
            if running >= target:
                return bucket
        return None

    def p99_latency(self) -> float | None:
        if not self._latency_count:
            return None
        target = self._latency_count * 0.99
        running = 0
        for bucket, count in self._latency_buckets.items():
            running += count
            if running >= target:
                return bucket
        return None

    def mean_latency(self) -> float | None:
        if not self._latency_count:
            return None
        return self._latency_sum / self._latency_count

    def to_prometheus(self) -> str:
        """Render all metrics in Prometheus text format."""
        lines = [
            "# HELP batch_agent_turns_total Total inference turns executed",
            "# TYPE batch_agent_turns_total counter",
            f"batch_agent_turns_total {self.turns_total}",
            "",
            "# HELP batch_agent_jobs_completed_total Jobs that completed successfully",
            "# TYPE batch_agent_jobs_completed_total counter",
            f"batch_agent_jobs_completed_total {self.jobs_completed}",
            "",
            "# HELP batch_agent_jobs_failed_total Jobs that failed after all retries",
            "# TYPE batch_agent_jobs_failed_total counter",
            f"batch_agent_jobs_failed_total {self.jobs_failed}",
            "",
            "# HELP batch_agent_cache_hit_rate Last observed prefix cache hit rate",
            "# TYPE batch_agent_cache_hit_rate gauge",
            f"batch_agent_cache_hit_rate {self.cache_hit_rate:.4f}",
            "",
            "# HELP batch_agent_concurrency_current Current max_concurrent setting",
            "# TYPE batch_agent_concurrency_current gauge",
            f"batch_agent_concurrency_current {self.concurrency_current}",
            "",
            "# HELP batch_agent_generate_latency_seconds Generate call latency",
            "# TYPE batch_agent_generate_latency_seconds histogram",
        ]
        for bucket, count in self._latency_buckets.items():
            le = "+Inf" if math.isinf(bucket) else str(bucket)
            lines.append(f'batch_agent_generate_latency_seconds_bucket{{le="{le}"}} {count}')
        lines.append(f"batch_agent_generate_latency_seconds_sum {self._latency_sum:.6f}")
        lines.append(f"batch_agent_generate_latency_seconds_count {self._latency_count}")
        lines.append("")
        return "\n".join(lines)


# ── HTTP metrics server (optional, started via start_metrics_server()) ─────────

_global_metrics: SchedulerMetrics | None = None


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = (_global_metrics.to_prometheus() if _global_metrics else "# no metrics\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass  # suppress access logs


def start_metrics_server(metrics: SchedulerMetrics, port: int = 9090) -> HTTPServer:
    """Start a background Prometheus metrics HTTP server on the given port."""
    global _global_metrics
    _global_metrics = metrics
    server = HTTPServer(("", port), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
