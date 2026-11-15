"""HTTP surface of the backend (Task 14): submit, poll, rate limit, errors.

``api.routes`` carries the two public scan endpoints (``POST /api/scan``,
``GET /api/scan/{id}``), ``api.serializers`` shapes the §7.6 payload by
recomputing the report from persisted rows, and ``api.ratelimit`` is the
sliding-window submission limiter keyed by salted-hashed client IP.
"""
