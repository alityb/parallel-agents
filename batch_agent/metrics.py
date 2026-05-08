from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricsSnapshot:
    completed: int = 0
    failed: int = 0
    queued: int = 0
