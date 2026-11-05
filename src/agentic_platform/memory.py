"""Session-scoped conversation memory backed by SQLite."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class SQLiteConversationMemory:
    """Store bounded conversation history separately for each session."""

    def __init__(self, database_path: str | Path, max_messages: int = 20) -> None:
        self.database_path = str(database_path)
        self.max_messages = max_messages
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_session
                ON conversation_memory(session_id, id)
                """
            )

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Persist a user/assistant turn and prune old messages for that session."""
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO conversation_memory (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                [
                    (session_id, "user", user_message),
                    (session_id, "assistant", assistant_message),
                ],
            )
            connection.execute(
                """
                DELETE FROM conversation_memory
                WHERE session_id = ? AND id NOT IN (
                    SELECT id FROM conversation_memory
                    WHERE session_id = ? ORDER BY id DESC LIMIT ?
                )
                """,
                (session_id, session_id, self.max_messages),
            )

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict[str, str]]:
        """Return messages in chronological order, isolated to one session."""
        count = limit or self.max_messages
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM conversation_memory
                    WHERE session_id = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, count),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear(self, session_id: str) -> int:
        """Delete all memory for a session and return the number of removed messages."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_memory WHERE session_id = ?",
                (session_id,),
            )
            return cursor.rowcount
