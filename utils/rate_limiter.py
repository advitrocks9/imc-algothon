"""Token-bucket rate limiter for exchange REST API."""
import threading
import time


class RateLimiter:
    """Enforces max N requests per second using a token bucket.

    Usage:
        limiter = RateLimiter(max_rps=1)
        limiter.acquire()  # blocks until a token is available
        # ... make request ...
    """

    def __init__(self, max_rps: float = 1.0):
        self._interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last_request = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_request = time.monotonic()
