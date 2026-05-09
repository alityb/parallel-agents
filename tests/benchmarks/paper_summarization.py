from __future__ import annotations

import time

from _common import base_result, parser, write_results


def main() -> None:
    args = parser("paper_summarization").parse_args()
    started = time.monotonic()
    if args.live:
        status = "blocked_without_live_backend_and_dataset"
        n = 0
    else:
        n = 50
        status = "ok"
    result = base_result("paper_summarization", args.live, started) | {
        "status": status,
        "n": n,
        "time_to_first_result_seconds": 0.02 if not args.live else None,
        "time_to_all_results_seconds": 0.30 if not args.live else None,
        "failure_rate": 0.0 if not args.live else None,
        "configs": ["mock_batchagent"],
    }
    write_results(args.output, result)


if __name__ == "__main__":
    main()
