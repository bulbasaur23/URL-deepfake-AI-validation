"""Submission rate limiter (Task 14): sliding window, POST-only scope, hashed keys.

``SlidingWindowRateLimiter`` bounds submissions per client: at most
``max_requests`` accepted requests per ``window_s`` sliding window, keyed by
the caller's salted IP hash. It is enforced at the ``POST /api/scan`` route
(ruling 3: the abuse vector is submissions — a 30/min GET cap would 429
legitimate 2 s polling of a 120 s scan), not as an ASGI middleware, so the
§7.5 URL validation can run BEFORE the limiter (ruling 5's flow: validate →
rate limit → submit) and invalid URLs never consume budget.

``hash_ip`` is the only bridge between a plaintext peer address and anything
the service keeps or keys on: ``sha256(settings.ip_salt + client_ip)`` — the
plaintext IP is never stored, logged, or used as a rate-limit key (spec GATE 4:
no plaintext IP storage). The same hash is what ``manager.submit`` persists as
``Scan.ip_hash``.

P1 scale note: the window state is in-memory per process (SQLite-local, like
the job manager's queue/warnings), protected by a lock so sync handlers
running in FastAPI's threadpool would be safe too; a multi-worker deployment
gets one limiter per worker.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque

from app.settings import Settings

# The hash format marker: sha256 hex digest (64 chars — matches Scan.ip_hash's
# String(64) column). Documented once, asserted never (it is an internal shape).


def hash_ip(client_ip: str, settings: Settings) -> str:
    """Salted sha256 of the client IP — the only acceptable key/storage form.

    ``sha256(settings.ip_salt + client_ip)`` per the ruling: the salt makes the
    digest non-reversible even for low-entropy IPs, and the plaintext address
    never touches disk, logs, or limiter keys.
    """
    return hashlib.sha256((settings.ip_salt + client_ip).encode("utf-8")).hexdigest()


class SlidingWindowRateLimiter:
    """Sliding-window counter: ≤ max_requests accepted per window_s, per key.

    Only ACCEPTED requests consume budget: a rejected (429) request is not
    appended, so a client retrying over the cap neither extends its window nor
    digs the hole deeper — it starts earning slots back as the window slides.
    Clock is ``time.monotonic`` (immune to wall-clock steps).
    """

    def __init__(self, window_s: float, max_requests: int) -> None:
        self._window_s = float(window_s)
        self._max_requests = int(max_requests)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record + admit one request for ``key`` (False = over the cap)."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            cutoff = now - self._window_s
            while hits and hits[0] <= cutoff:
                hits.popleft()  # slide: drop entries older than the window
            if len(hits) >= self._max_requests:
                return False  # rejected requests do not consume budget
            hits.append(now)
            return True
