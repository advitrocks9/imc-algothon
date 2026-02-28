"""Local inventory manager with periodic cloud reconciliation.

Maintains local position state updated optimistically from fills.
Only falls back to the exchange REST API on startup, every 60 seconds,
or when an error forces a dirty sync.
"""

import logging
import threading
import time

from bot_template import Side

log = logging.getLogger("inventory")

CLOUD_SYNC_INTERVAL = 60.0  # seconds between cloud reconciliations


class InventoryManager:
    """Thread-safe local position tracker.

    The main loop, SSE callbacks, and executor threads may all call
    methods concurrently — all public methods are lock-protected.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._positions: dict[str, int] = {}
        self._last_cloud_sync: float = 0.0
        self._initialized = False
        self._dirty = False

    def initialize(self, cloud_positions: dict[str, int]) -> None:
        """Set initial positions from REST call. Call once at startup."""
        with self._lock:
            self._positions = dict(cloud_positions)
            self._last_cloud_sync = time.monotonic()
            self._initialized = True
            self._dirty = False
        log.info(f"Inventory initialized: {self._positions}")

    def apply_fill(self, product: str, side: Side, filled_volume: int) -> None:
        """Update local position after a confirmed fill.

        Args:
            product: Symbol that was filled.
            side: Side.BUY or Side.SELL.
            filled_volume: Number of contracts filled.
        """
        if filled_volume <= 0:
            return
        with self._lock:
            current = self._positions.get(product, 0)
            if side == Side.BUY:
                self._positions[product] = current + filled_volume
            else:
                self._positions[product] = current - filled_volume
            log.debug(
                f"Local fill: {side} {filled_volume}x {product} "
                f"-> pos={self._positions[product]}"
            )

    def apply_fills_from_report(self, leg_results: list) -> None:
        """Batch-update positions from an ExecutionReport's leg results.

        Args:
            leg_results: list of LegResult from ConcurrentExecutor.
        """
        for lr in leg_results:
            if lr.filled > 0:
                self.apply_fill(lr.order.product, lr.order.side, lr.filled)

    @property
    def positions(self) -> dict[str, int]:
        """Return a copy of current local positions."""
        with self._lock:
            return dict(self._positions)

    def get_position(self, product: str) -> int:
        with self._lock:
            return self._positions.get(product, 0)

    def needs_cloud_sync(self) -> bool:
        """True if enough time has elapsed or state is dirty."""
        with self._lock:
            return (
                self._dirty
                or not self._initialized
                or (time.monotonic() - self._last_cloud_sync) >= CLOUD_SYNC_INTERVAL
            )

    def cloud_sync(self, cloud_positions: dict[str, int]) -> dict[str, int]:
        """Reconcile local state with cloud. Returns drift (cloud - local).

        The cloud is authoritative. Any drift is logged as a warning.
        """
        with self._lock:
            drift: dict[str, int] = {}
            all_symbols = set(list(self._positions.keys()) + list(cloud_positions.keys()))
            for symbol in all_symbols:
                local_val = self._positions.get(symbol, 0)
                cloud_val = cloud_positions.get(symbol, 0)
                if local_val != cloud_val:
                    drift[symbol] = cloud_val - local_val
            if drift:
                log.warning(f"Position drift detected: {drift}")
                log.warning(f"  Local:  {self._positions}")
                log.warning(f"  Cloud:  {cloud_positions}")
            self._positions = dict(cloud_positions)
            self._last_cloud_sync = time.monotonic()
            self._dirty = False
            return drift

    def mark_dirty(self) -> None:
        """Force a cloud sync on the next needs_cloud_sync() check."""
        with self._lock:
            self._dirty = True
