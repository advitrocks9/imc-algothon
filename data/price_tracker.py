"""SSE-fed price tracker with EMA signals and volatility.

Consumes mid-price updates from on_orderbook() (free SSE stream)
and provides trading signals for all strategies.
"""
import math
import threading
import time
from collections import defaultdict


class PriceTracker:
    """Maintains EMA price history and trading signals from SSE mid-price updates.

    Called from on_orderbook() on the SSE thread. All reads from the main loop.
    Thread-safe via a single lock (critical section is tiny).
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        self.fast_period = fast_period
        self.slow_period = slow_period

        self._lock = threading.Lock()

        # Per-symbol state
        self._mid_prices: dict[str, list[float]] = defaultdict(list)
        self._ema_fast: dict[str, float | None] = defaultdict(lambda: None)
        self._ema_slow: dict[str, float | None] = defaultdict(lambda: None)

        # Derived
        self._signal: dict[str, float] = defaultdict(float)
        self._volatility: dict[str, float] = defaultdict(float)

    # --- Writer (SSE thread) ---

    def update(self, symbol: str, mid: float) -> None:
        """Called from on_orderbook when a new mid price is available."""
        if mid is None or math.isnan(mid):
            return

        with self._lock:
            prices = self._mid_prices[symbol]
            prices.append(mid)

            # Cap memory at 200 prices
            if len(prices) > 200:
                self._mid_prices[symbol] = prices[-200:]

            self._update_ema(symbol, mid)
            self._update_volatility(symbol)
            self._update_signal(symbol)

    def _update_ema(self, symbol: str, price: float) -> None:
        alpha_fast = 2.0 / (self.fast_period + 1)
        if self._ema_fast[symbol] is None:
            self._ema_fast[symbol] = price
        else:
            self._ema_fast[symbol] = alpha_fast * price + (1 - alpha_fast) * self._ema_fast[symbol]

        alpha_slow = 2.0 / (self.slow_period + 1)
        if self._ema_slow[symbol] is None:
            self._ema_slow[symbol] = price
        else:
            self._ema_slow[symbol] = alpha_slow * price + (1 - alpha_slow) * self._ema_slow[symbol]

    def _update_volatility(self, symbol: str) -> None:
        prices = self._mid_prices[symbol]
        if len(prices) < 3:
            return
        recent = prices[-20:]
        returns = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        if not returns:
            return
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        self._volatility[symbol] = math.sqrt(var) if var > 0 else 0.0

    def _update_signal(self, symbol: str) -> None:
        fast = self._ema_fast[symbol]
        slow = self._ema_slow[symbol]
        if fast is None or slow is None or slow == 0:
            self._signal[symbol] = 0.0
            return

        # Normalized crossover signal clamped to [-1, 1]
        raw = (fast - slow) / abs(slow) * 100
        self._signal[symbol] = max(-1.0, min(1.0, raw))

    # --- Readers (main loop thread) ---

    def get_signal(self, symbol: str) -> float:
        """Signal in [-1, 1]. Positive = bullish."""
        with self._lock:
            return self._signal[symbol]

    def get_ema_fast(self, symbol: str) -> float | None:
        with self._lock:
            return self._ema_fast[symbol]

    def get_ema_slow(self, symbol: str) -> float | None:
        with self._lock:
            return self._ema_slow[symbol]

    def get_mid(self, symbol: str) -> float | None:
        with self._lock:
            prices = self._mid_prices[symbol]
            return prices[-1] if prices else None

    def get_volatility(self, symbol: str) -> float:
        with self._lock:
            return self._volatility[symbol]

    def get_price_count(self, symbol: str) -> int:
        with self._lock:
            return len(self._mid_prices[symbol])

    def warmup_complete(self, symbol: str, min_ticks: int = 10) -> bool:
        with self._lock:
            return len(self._mid_prices[symbol]) >= min_ticks
