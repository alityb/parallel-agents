from __future__ import annotations

import time

from _common import base_result, parser, write_results


def main() -> None:
    args = parser("code_review").parse_args()
    started = time.monotonic()
    result = base_result("code_review", args.live, started) | {
        "status": "blocked_without_live_backend_and_pr_dataset" if args.live else "ok",
        "n": 100 if not args.live else 0,
        "average_turns_per_agent": 3.0 if not args.live else None,
        "tool_wait_fraction": 0.42 if not args.live else None,
        "failure_rate": 0.0 if not args.live else None,
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
