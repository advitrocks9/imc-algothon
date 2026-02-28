"""Main trading bot."""
import threading
import time
import logging

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Trade, Side
from config import (
    EXCHANGE_URL, USERNAME, PASSWORD, SYMBOLS, MIN_ARB_EDGE,
    MAX_SCRATCH_COST,
)
from execution.arbitrage import ArbitrageEngine
from execution.executor import AsyncExecutor
from execution.inventory import InventoryManager
from execution.scratch import ScratchRoutine
from risk.manager import RiskManager
from utils.rate_limiter import RateLimiter
from utils.helpers import best_bid, best_ask, best_bid_with_volume, best_ask_with_volume, mid_price, snap_to_tick

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot")


class TradingBot(BaseBot):
    def __init__(self):
        super().__init__(EXCHANGE_URL, USERNAME, PASSWORD)
        self.risk = RiskManager()
        self.limiter = RateLimiter(max_rps=1.0)
        self.arb_engine = ArbitrageEngine(min_edge=MIN_ARB_EDGE)

        # Concurrent execution, local inventory, scratch routine
        self.executor = AsyncExecutor(cmi_url=self._cmi_url, auth_token=self.auth_token)
        self.inventory = InventoryManager()
        self.scratch = ScratchRoutine(aggression_ticks=2)

        # Latest order book snapshots (updated by SSE)
        self._books_lock = threading.Lock()
        self._books: dict[str, OrderBook] = {}
        self._book_event = threading.Event()


    # --- SSE Callbacks ---

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Cache latest book snapshot and wake the main loop."""
        with self._books_lock:
            self._books[orderbook.product] = orderbook
        self._book_event.set()

    def on_trades(self, trade: Trade) -> None:
        side = "BOUGHT" if trade.buyer == self.username else "SOLD"
        log.info(f"FILL: {side} {trade.volume}x {trade.product} @ {trade.price}")

    def get_book(self, symbol: str) -> OrderBook | None:
        with self._books_lock:
            return self._books.get(symbol)

    # --- Rate-Limited REST Wrappers ---

    def safe_get_positions(self) -> dict[str, int]:
        self.limiter.acquire()
        return self.get_positions()

    def safe_get_orderbook(self, symbol: str) -> OrderBook:
        self.limiter.acquire()
        return self.get_orderbook(symbol)

    def safe_send_order(self, order: OrderRequest) -> OrderResponse | None:
        self.limiter.acquire()
        return self.send_order(order)

    def safe_cancel_order(self, order_id: str) -> None:
        self.limiter.acquire()
        self.cancel_order(order_id)

    def safe_get_orders(self, product: str | None = None) -> list[dict]:
        self.limiter.acquire()
        return self.get_orders(product)

    def safe_cancel_all_orders(self) -> None:
        """Cancel all orders, one at a time through rate limiter."""
        orders = self.safe_get_orders()
        for o in orders:
            self.safe_cancel_order(o["id"])

    def safe_send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        """IOC: place + cancel. Costs 2 rate-limited requests."""
        resp = self.safe_send_order(order)
        if resp and resp.volume > 0:
            self.safe_cancel_order(resp.id)
        return resp

    # --- Main Loop ---

    def run(self) -> None:
        log.info("Starting bot...")
        self.start()
        log.info("SSE stream connected.")

        # Initial position sync — one-time REST call
        cloud_positions = self.safe_get_positions()
        self.inventory.initialize(cloud_positions)
        self.risk.update_positions(cloud_positions)
        log.info(f"Positions: {self.inventory.positions}")

        # List products
        self.limiter.acquire()
        products = {p.symbol: p for p in self.get_products()}
        log.info(f"Products: {list(products.keys())}")

        try:
            while True:
                self._book_event.wait(timeout=60)
                self._book_event.clear()
                self._main_tick(products)
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self.executor.shutdown()
            self.stop()

    def _main_tick(self, products: dict) -> None:
        """One iteration of the main trading loop."""
        # 1. Read cached SSE books (no REST call)
        with self._books_lock:
            books = dict(self._books)

        # 2. Check ETF arbitrage using LOCAL positions
        arb_orders = self.arb_engine.check_and_generate_orders(
            books, self.inventory.positions, max_volume=5
        )

        if arb_orders:
            # 3. Execute all legs concurrently
            report = self.executor.execute_ioc_batch(arb_orders)

            # 4. Update local inventory + risk from fills
            self.inventory.apply_fills_from_report(report.legs)
            self.risk.update_positions(self.inventory.positions)

            for lr in report.legs:
                log.info(
                    f"ARB leg: {lr.order.product} {lr.order.side} "
                    f"filled={lr.filled} @ {lr.order.price}"
                    + (f" ERROR={lr.error}" if lr.error else "")
                )

            # 5. Scratch if partial fill
            if self.scratch.needs_scratch(report):
                self._execute_scratch(report)

        # 6. Periodic cloud sync (every 60s, not every tick)
        if self.inventory.needs_cloud_sync():
            self._cloud_sync()

    def _execute_scratch(self, report) -> None:
        """Handle partial fills by reversing excess positions."""
        # Re-read books for current market prices
        with self._books_lock:
            fresh_books = dict(self._books)

        scratch_plan = self.scratch.plan_scratch(report, fresh_books)
        if scratch_plan is None:
            return

        # Skip scratch if estimated cost is too high — just sync positions instead
        if scratch_plan.expected_cost > MAX_SCRATCH_COST:
            log.warning(
                f"SCRATCH SKIPPED: estimated cost={scratch_plan.expected_cost:.2f} "
                f"exceeds cap={MAX_SCRATCH_COST}. Forcing cloud sync instead."
            )
            self.inventory.mark_dirty()
            return

        log.warning(
            f"SCRATCH: Reversing {scratch_plan.excess_by_leg}, "
            f"estimated cost={scratch_plan.expected_cost:.2f}"
        )

        # Execute scratch orders concurrently
        scratch_report = self.executor.execute_ioc_batch(scratch_plan.orders)

        # Update inventory from scratch fills
        self.inventory.apply_fills_from_report(scratch_report.legs)
        self.risk.update_positions(self.inventory.positions)

        # Log scratch results
        for lr in scratch_report.legs:
            log.warning(
                f"SCRATCH leg: {lr.order.product} {lr.order.side} "
                f"filled={lr.filled}/{lr.order.volume} @ {lr.order.price}"
                + (f" ERROR={lr.error}" if lr.error else "")
            )

        unfilled_scratch = [
            lr for lr in scratch_report.legs if lr.filled < lr.order.volume
        ]
        if unfilled_scratch:
            log.error(
                f"SCRATCH INCOMPLETE: {len(unfilled_scratch)} legs not fully filled. "
                f"Residual risk remains — forcing cloud sync."
            )

        # Force cloud sync to reconcile after scratch
        self.inventory.mark_dirty()

    def _cloud_sync(self) -> None:
        """Reconcile local positions with the exchange."""
        try:
            cloud_pos = self.safe_get_positions()
            drift = self.inventory.cloud_sync(cloud_pos)
            self.risk.update_positions(cloud_pos)
            if drift:
                log.warning(f"Position drift corrected: {drift}")
        except Exception as e:
            log.error(f"Cloud sync failed: {e}")
            self.inventory.mark_dirty()
