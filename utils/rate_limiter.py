"""Token-bucket rate limiter for exchange REST API.

Supports N requests per second with burst capability.
Thread-safe: multiple threads can acquire tokens concurrently.
"""
import threading
import time


class RateLimiter:
    """Token bucket allowing up to max_rps requests per second.

    Tokens refill continuously. Supports burst up to max_rps tokens
    when the bucket is full, then drains at 1 token per (1/max_rps) seconds.

    Usage:
        limiter = RateLimiter(max_rps=8)
        limiter.acquire()  # blocks until a token is available
        # ... make request ...
    """

    def __init__(self, max_rps: float = 8.0):
        self._max_tokens = max_rps
        self._refill_rate = max_rps  # tokens per second
        self._tokens = max_rps      # start full
        self._lock = threading.Lock()
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) / self._refill_rate
            time.sleep(wait)

    def _refill(self) -> None:
        """Add tokens based on elapsed time. Must be called under lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now
