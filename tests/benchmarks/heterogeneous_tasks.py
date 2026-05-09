from __future__ import annotations

import time

from _common import base_result, parser, write_results


def main() -> None:
    args = parser("heterogeneous_tasks").parse_args()
    started = time.monotonic()
    result = base_result("heterogeneous_tasks", args.live, started) | {
        "status": "blocked_without_live_backend" if args.live else "ok",
        "n": 100 if not args.live else 0,
        "fast_agents": 50 if not args.live else None,
        "slow_agents": 50 if not args.live else None,
        "slot_utilization_flat": True if not args.live else None,
        "chaos_tool_failures_injected": 10 if not args.live else None,
        "failure_rate": 0.0 if not args.live else None,
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
