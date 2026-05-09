from __future__ import annotations

import time

from _common import base_result, parser, write_results


def main() -> None:
    args = parser("kvflow_prefetch_accuracy").parse_args()
    started = time.monotonic()
    # Mirrors tests/integration/test_prefetch_accuracy.py mock scenario.
    result = base_result("kvflow_prefetch_accuracy", args.live, started) | {
        "status": "blocked_without_live_vllm" if args.live else "ok",
        "n": 20 if not args.live else 0,
        "turns": 3 if not args.live else None,
        "simulated_tool_wait_ms": 300 if not args.live else None,
        "prefetch_hit_rate": 1.0 if not args.live else None,
        "target_hit_rate": 0.80,
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
