"""Rate-limit-efficient order scheduling and resting order management.

OrderScheduler: weighted round-robin giving Group A 4x priority.
RestingOrderManager: tracks GTC orders, only reprices when needed.
"""
import time
from collections import Counter

from config import GROUP_A

NON_GROUP_A = ["TIDE_SWING", "WX_SUM", "LHR_INDEX", "LON_FLY"]


class OrderScheduler:
    """Weighted round-robin across all 8 symbols.

    Group A (4 symbols x 4 slots) = 16 slots
    Others  (4 symbols x 1 slot)  =  4 slots
    Total = 20 slots per cycle.
    At 1 req/sec ≈ Group A refreshed every ~5s, others every ~20s.
    """

    def __init__(self):
        raw: list[str] = []
        for sym in GROUP_A:
            raw.extend([sym] * 4)
        for sym in NON_GROUP_A:
            raw.append(sym)

        self._queue = self._interleave(raw)
        self._index = 0

    def next_symbol(self) -> str:
        sym = self._queue[self._index % len(self._queue)]
        self._index += 1
        return sym

    @staticmethod
    def _interleave(queue: list[str]) -> list[str]:
        """Spread symbols evenly to avoid consecutive duplicates."""
        counts = dict(Counter(queue))
        result: list[str] = []
        while any(v > 0 for v in counts.values()):
            for sym in sorted(counts, key=lambda s: -counts[s]):
                if counts[sym] > 0:
                    result.append(sym)
                    counts[sym] -= 1
        return result


class RestingOrderManager:
    """Tracks our resting GTC orders per symbol.

    Only triggers cancel+replace when price drifts beyond threshold,
    saving REST requests.
    """

    def __init__(self, reprice_threshold: float = 3.0, stale_seconds: float = 120.0):
        self._reprice_threshold = reprice_threshold
        self._stale_seconds = stale_seconds
        # symbol -> {"bid_id", "ask_id", "bid_price", "ask_price", "bid_time", "ask_time"}
        self._orders: dict[str, dict] = {}

    def needs_update(
        self, symbol: str, new_bid: float | None, new_ask: float | None
    ) -> tuple[bool, bool]:
        """Returns (need_update_bid, need_update_ask)."""
        current = self._orders.get(symbol)
        if current is None:
            return (new_bid is not None, new_ask is not None)

        now = time.monotonic()
        need_bid = False
        need_ask = False

        if new_bid is not None:
            old_bid = current.get("bid_price")
            bid_time = current.get("bid_time", 0)
            if old_bid is None:
                need_bid = True
            elif abs(new_bid - old_bid) >= self._reprice_threshold:
                need_bid = True
            elif (now - bid_time) > self._stale_seconds:
                need_bid = True

        if new_ask is not None:
            old_ask = current.get("ask_price")
            ask_time = current.get("ask_time", 0)
            if old_ask is None:
                need_ask = True
            elif abs(new_ask - old_ask) >= self._reprice_threshold:
                need_ask = True
            elif (now - ask_time) > self._stale_seconds:
                need_ask = True

        return (need_bid, need_ask)

    def record_order(self, symbol: str, side: str, order_id: str, price: float) -> None:
        if symbol not in self._orders:
            self._orders[symbol] = {}
        now = time.monotonic()
        if side == "BUY":
            self._orders[symbol]["bid_id"] = order_id
            self._orders[symbol]["bid_price"] = price
            self._orders[symbol]["bid_time"] = now
        else:
            self._orders[symbol]["ask_id"] = order_id
            self._orders[symbol]["ask_price"] = price
            self._orders[symbol]["ask_time"] = now

    def get_order_id(self, symbol: str, side: str) -> str | None:
        current = self._orders.get(symbol)
        if current is None:
            return None
        return current.get("bid_id" if side == "BUY" else "ask_id")

    def clear_order(self, symbol: str, side: str) -> None:
        """Remove a tracked order (e.g. after cancel or fill)."""
        current = self._orders.get(symbol)
        if current is None:
            return
        key = "bid" if side == "BUY" else "ask"
        current.pop(f"{key}_id", None)
        current.pop(f"{key}_price", None)
        current.pop(f"{key}_time", None)
