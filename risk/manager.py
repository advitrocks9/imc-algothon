"""Position and risk management."""
import threading

from config import MAX_POSITION


class RiskManager:
    """Tracks positions and enforces limits.

    Thread-safe. Updated by the main loop after fills and periodically
    reconciled against the exchange via get_positions().
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._positions: dict[str, int] = {}
        self._pnl: dict = {}

    def update_positions(self, positions: dict[str, int]) -> None:
        with self._lock:
            self._positions = dict(positions)

    def get_position(self, symbol: str) -> int:
        with self._lock:
            return self._positions.get(symbol, 0)

    def can_buy(self, symbol: str, volume: int) -> bool:
        return self.get_position(symbol) + volume <= MAX_POSITION

    def can_sell(self, symbol: str, volume: int) -> bool:
        return self.get_position(symbol) - volume >= -MAX_POSITION

    def max_buy_volume(self, symbol: str) -> int:
        return max(0, MAX_POSITION - self.get_position(symbol))

    def max_sell_volume(self, symbol: str) -> int:
        return max(0, MAX_POSITION + self.get_position(symbol))

    def update_pnl(self, pnl: dict) -> None:
        with self._lock:
            self._pnl = pnl

    @property
    def positions(self) -> dict[str, int]:
        with self._lock:
            return dict(self._positions)
