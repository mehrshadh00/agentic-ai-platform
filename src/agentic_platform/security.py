"""Small, dependency-free API security primitives."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict
from fastapi import HTTPException, status


class FixedWindowRateLimiter:
    """Thread-safe per-client fixed-window limiter for a single API process."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.requests_per_minute = requests_per_minute
        self._windows: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float | None = None) -> tuple[bool, int]:
        current = time.time() if now is None else now
        window = int(current // 60)
        with self._lock:
            previous_window, count = self._windows[client_id]
            if previous_window != window:
                count = 0
            if count >= self.requests_per_minute:
                retry_after = max(1, int(60 - (current % 60)))
                self._windows[client_id] = (window, count)
                return False, retry_after
            self._windows[client_id] = (window, count + 1)
            return True, 0

    def reset(self) -> None:
        """Clear counters; useful for deterministic tests and controlled resets."""
        with self._lock:
            self._windows.clear()


def verify_api_key(expected: str | None, supplied: str | None) -> bool:
    """Allow local development without a configured key; compare configured keys safely."""
    if not expected:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(expected.encode(), supplied.encode())


