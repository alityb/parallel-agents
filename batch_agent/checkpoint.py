"""SQLite-based checkpoint store for agent state persistence.

Provides crash recovery: if the orchestration process dies mid-run,
re-running with the same checkpoint_dir skips already-completed agents.
Also serves as overflow when in-process state store exceeds memory limits.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from .spec import AgentError, AgentResult, Message
from .state import AgentState, AgentStatus

logger = logging.getLogger(__name__)


class CheckpointStore:
    """Persists agent state to SQLite for crash recovery."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.path = Path(checkpoint_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / "checkpoint.db"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS agent_states (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                turn INTEGER NOT NULL DEFAULT 0,
                messages TEXT NOT NULL DEFAULT '[]',
                output TEXT,
                error TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_updated REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                job_id TEXT PRIMARY KEY,
                idx INTEGER NOT NULL,
                output TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 1
            );
        """)
        self._conn.commit()

    def save_state(self, state: AgentState) -> None:
        """Save agent state after each turn."""
        messages_json = json.dumps([{"role": m.role, "content": m.content} for m in state.messages])
        error_json = json.dumps(state.error.__dict__) if state.error else None
        output_json = json.dumps(state.output, default=str) if state.output is not None else None

        self._conn.execute("""
            INSERT OR REPLACE INTO agent_states
            (job_id, status, turn, messages, output, error, retry_count, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.job_id,
            state.status.value,
            state.turn,
            messages_json,
            output_json,
            error_json,
            state.retry_count,
            state.created_at,
            state.last_updated,
        ))
        self._conn.commit()

    def save_result(self, result: AgentResult) -> None:
        """Save a completed result."""
        output_json = None
        if result.output is not None:
            if hasattr(result.output, "model_dump"):
                output_json = json.dumps(result.output.model_dump())
            elif hasattr(result.output, "dict"):
                output_json = json.dumps(result.output.dict())
            else:
                output_json = json.dumps(result.output, default=str)

        error_json = json.dumps(result.error.__dict__) if result.error else None

        self._conn.execute("""
            INSERT OR REPLACE INTO results (job_id, idx, output, error, attempts)
            VALUES (?, ?, ?, ?, ?)
        """, (result.job_id, result.index, output_json, error_json, result.attempts))
        self._conn.commit()

    def get_completed_job_ids(self) -> set[str]:
        """Return set of job_ids that have already completed."""
        cursor = self._conn.execute("SELECT job_id FROM results")
        return {row[0] for row in cursor.fetchall()}

    def load_result(self, job_id: str) -> AgentResult | None:
        """Load a previously completed result."""
        cursor = self._conn.execute(
            "SELECT job_id, idx, output, error, attempts FROM results WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        output = json.loads(row[2]) if row[2] else None
        error_dict = json.loads(row[3]) if row[3] else None
        error = AgentError(**error_dict) if error_dict else None

        return AgentResult(
            job_id=row[0],
            index=row[1],
            output=output,
            error=error,
            attempts=row[4],
        )

    def load_state(self, job_id: str) -> AgentState | None:
        """Load the last checkpointed in-progress agent state."""
        cursor = self._conn.execute(
            """
            SELECT job_id, status, turn, messages, output, error, retry_count, created_at, last_updated
            FROM agent_states WHERE job_id = ?
            """,
            (job_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        messages_raw = json.loads(row[3]) if row[3] else []
        messages = [Message(role=m["role"], content=m["content"]) for m in messages_raw]
        output = json.loads(row[4]) if row[4] else None
        error_dict = json.loads(row[5]) if row[5] else None
        error = AgentError(**error_dict) if error_dict else None

        return AgentState(
            job_id=row[0],
            status=AgentStatus(row[1]),
            turn=row[2],
            messages=messages,
            output=output,
            error=error,
            retry_count=row[6],
            created_at=row[7],
            last_updated=row[8],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
