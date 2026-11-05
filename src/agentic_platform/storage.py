"""SQLite persistence for task responses and their execution traces."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agentic_platform.models import TaskResponse, TraceEvent


class SQLiteTraceStore:
    """Persist complete task reports and individually queryable trace events."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_events (
                    task_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (task_id, sequence),
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                )
                """
            )

    def save_response(self, response: TaskResponse) -> None:
        """Atomically save a task report and its ordered trace events."""
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT OR REPLACE INTO tasks (task_id, status, response_json)
                VALUES (?, ?, ?)
                """,
                (response.task_id, response.status, response.model_dump_json()),
            )
            connection.execute("DELETE FROM trace_events WHERE task_id = ?", (response.task_id,))
            connection.executemany(
                """
                INSERT INTO trace_events (task_id, sequence, event_json)
                VALUES (?, ?, ?)
                """,
                [
                    (response.task_id, index, event.model_dump_json())
                    for index, event in enumerate(response.trace)
                ],
            )

    def get_response(self, task_id: str) -> TaskResponse | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return TaskResponse.model_validate_json(row["response_json"])

    def get_trace(self, task_id: str) -> list[TraceEvent] | None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if exists is None:
                return None
            rows = connection.execute(
                """
                SELECT event_json FROM trace_events
                WHERE task_id = ? ORDER BY sequence
                """,
                (task_id,),
            ).fetchall()
        return [TraceEvent.model_validate_json(row["event_json"]) for row in rows]
