"""Main trading bot."""
import os
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Trade, Side
from config import (
    EXCHANGE_URL, USERNAME, PASSWORD, SYMBOLS, GROUP_A, MIN_ARB_EDGE,
    MAX_SCRATCH_COST, ARB_COOLDOWN_SECONDS, STRATEGY_WARMUP_TICKS,
    REPRICE_THRESHOLD, STALE_ORDER_SECONDS, AGGRESSIVE_THRESHOLD,
    AGGRESSIVE_ORDER_SIZE, MAX_POSITION,
)
from data.price_tracker import PriceTracker
from execution.arbitrage import ArbitrageEngine
from execution.executor import AsyncExecutor
from execution.inventory import InventoryManager
from execution.order_scheduler import OrderScheduler, RestingOrderManager
from execution.scratch import ScratchRoutine
from execution.strategies import (
    STRATEGY_CONFIGS, compute_desired_orders, compute_aggressive_ioc,
)
from risk.manager import RiskManager
from theo.engine import TheoEngine
from utils.rate_limiter import RateLimiter
from utils.helpers import mid_price, best_ask, best_bid

# --- Logging setup: console + file in logs/ ---
os.makedirs("logs", exist_ok=True)
_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/bot_{_timestamp}.log"),
    ],
)
log = logging.getLogger("bot")


class TradingBot(BaseBot):
    def __init__(self):
        super().__init__(EXCHANGE_URL, USERNAME, PASSWORD)
        self.risk = RiskManager()
        self.limiter = RateLimiter(max_rps=16.0)
        self.arb_engine = ArbitrageEngine(min_edge=MIN_ARB_EDGE)

        # Concurrent execution, local inventory, scratch routine
        self.executor = AsyncExecutor(cmi_url=self._cmi_url, auth_token=self.auth_token)
        self.inventory = InventoryManager()
        self.scratch = ScratchRoutine(aggression_ticks=2)

        # Latest order book snapshots (updated by SSE)
        self._books_lock = threading.Lock()
        self._books: dict[str, OrderBook] = {}
        self._book_event = threading.Event()

        # Strategy components
        self.price_tracker = PriceTracker(fast_period=5, slow_period=20)
        self.order_scheduler = OrderScheduler()
        self.resting_orders = RestingOrderManager(
            reprice_threshold=REPRICE_THRESHOLD,
            stale_seconds=STALE_ORDER_SECONDS,
        )
        self._arb_cooldown_until = 0.0
        self._last_pnl_log = 0.0
        self._last_aggressive_tick = 0.0
        self._aggressive_start_idx = 0  # round-robin for aggressive IOC
        self._aggressive_misses: dict[str, int] = {}  # consecutive unfilled attempts
        self._aggressive_backoff_until: dict[str, float] = {}  # backoff expiry per symbol

        # Theo engine — computes fair values from external data
        self.theo_engine = TheoEngine()

        # Thread pool for parallel order dispatch (8 req/sec budget)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="order")

        # Emergency unwind tracking
        self._last_emergency_unwind = 0.0


    # --- SSE Callbacks ---

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Cache latest book snapshot, feed price tracker, and wake main loop."""
        with self._books_lock:
            self._books[orderbook.product] = orderbook

        # Feed mid price to tracker (SSE is free — no rate limit cost)
        mid = mid_price(orderbook)
        if mid is not None:
            self.price_tracker.update(orderbook.product, mid)

        self._book_event.set()

    def on_trades(self, trade: Trade) -> None:
        if trade.buyer == self.username:
            log.info(f"OWN FILL: BOUGHT {trade.volume}x {trade.product} @ {trade.price}")
            self.inventory.apply_fill(trade.product, Side.BUY, trade.volume)
            self.risk.update_positions(self.inventory.positions)
            self.inventory.mark_dirty()
        elif trade.seller == self.username:
            log.info(f"OWN FILL: SOLD {trade.volume}x {trade.product} @ {trade.price}")
            self.inventory.apply_fill(trade.product, Side.SELL, trade.volume)
            self.risk.update_positions(self.inventory.positions)
            self.inventory.mark_dirty()

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

    def safe_get_pnl(self) -> dict:
        self.limiter.acquire()
        return self.get_pnl()

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

        # Start theo engine (background thread for external data)
        self.theo_engine.start()
        log.info("Theo engine started.")

        # Initial position sync — one-time REST call
        cloud_positions = self.safe_get_positions()
        self.inventory.initialize(cloud_positions)
        self.risk.update_positions(cloud_positions)
        log.info(f"Inventory initialized: {self.inventory.positions}")
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
            self.theo_engine.stop()
            self.executor.shutdown()
            self._pool.shutdown(wait=False)
            self.stop()

    def _main_tick(self, products: dict) -> None:
        """One iteration of the main trading loop."""
        # 1. Read cached SSE books (no REST call)
        with self._books_lock:
            books = dict(self._books)

        # 1b. Emergency position unwinding for limit-blocked Group A products
        self._emergency_unwind_tick(books)

        # 2. Check ETF arbitrage using LOCAL positions
        arb_orders = self.arb_engine.check_and_generate_orders(
            books, self.inventory.positions, max_volume=2
        )

        if arb_orders:
            # 3. Execute all legs concurrently
            report = self.executor.execute_ioc_batch(arb_orders)

            # 4. Update risk from fills (DON'T apply_fills here — on_trades SSE
            #    callback handles it, and double-counting causes position drift)
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

            # 6. Arb cooldown — pause strategy orders to avoid 429s
            self._arb_cooldown_until = time.monotonic() + ARB_COOLDOWN_SECONDS
            return

        # 7. Run strategy-based quoting (only if not in arb cooldown)
        if time.monotonic() >= self._arb_cooldown_until:
            # 7a. Aggressive IOC on mispriced Group A products (every 1s with 16 req/sec)
            now = time.monotonic()
            if now - self._last_aggressive_tick >= 1.0:
                self._aggressive_tick(books)
                self._last_aggressive_tick = now

            # 7b. Passive strategy quoting (one symbol per tick)
            self._strategy_tick(books)

        # 8. Periodic PnL logging (every 60s to save API budget)
        now = time.monotonic()
        if now - self._last_pnl_log >= 60:
            self._log_pnl()
            self._last_pnl_log = now

        # 9. Periodic cloud sync (every 60s, not every tick)
        if self.inventory.needs_cloud_sync():
            self._cloud_sync()

    # --- Emergency Unwind ---

    def _emergency_unwind_tick(self, books: dict[str, OrderBook]) -> None:
        """Aggressively unwind Group A products that are near position limit.

        When a component product (especially LHR_COUNT) is at the limit, the
        arb engine is completely blocked. Buy/sell to free headroom.
        """
        now = time.monotonic()
        if now - self._last_emergency_unwind < 1.0:
            return  # only attempt every 1 second (16 req/sec budget)

        UNWIND_THRESHOLD = 60  # start unwinding when abs(pos) >= this
        COMPONENT_SYMBOLS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]

        for symbol in COMPONENT_SYMBOLS:
            pos = self.inventory.get_position(symbol)
            if abs(pos) < UNWIND_THRESHOLD:
                continue

            book = books.get(symbol)
            if book is None:
                continue

            # Determine unwind direction
            if pos >= UNWIND_THRESHOLD:
                # Too long — sell to reduce
                target_price = best_bid(book)
                if target_price is None:
                    continue
                side = Side.SELL
                size = min(10, pos - 30)  # try to get back to ~30
            elif pos <= -UNWIND_THRESHOLD:
                # Too short — buy to reduce
                target_price = best_ask(book)
                if target_price is None:
                    continue
                side = Side.BUY
                size = min(10, abs(pos) - 30)  # try to get back to ~-30
            else:
                continue

            if size <= 0:
                continue

            log.warning(
                f"EMERGENCY UNWIND: {symbol} {side.name} {size}x @ {target_price} "
                f"(pos={pos}, freeing arb headroom)"
            )
            order = OrderRequest(symbol, target_price, side, size)
            self.safe_send_ioc(order)
            self._last_emergency_unwind = now
            return  # one unwind per tick to stay within rate limit

    # --- Strategy Engine ---

    def _strategy_tick(self, books: dict[str, OrderBook]) -> None:
        """Place/update GTC resting orders for ALL symbols in parallel."""
        futures = []
        for symbol in SYMBOLS:
            if not self.price_tracker.warmup_complete(symbol, STRATEGY_WARMUP_TICKS):
                continue

            position = self.inventory.get_position(symbol)
            bid_price, ask_price, bid_size, ask_size = compute_desired_orders(
                symbol, self.price_tracker, position,
                theo_engine=self.theo_engine,
            )

            need_bid, need_ask = self.resting_orders.needs_update(
                symbol, bid_price, ask_price,
            )

            if need_bid or need_ask:
                fut = self._pool.submit(
                    self._reconcile_orders,
                    symbol, bid_price, ask_price, bid_size, ask_size,
                    need_bid, need_ask,
                )
                futures.append(fut)

        # Wait for all parallel order updates to complete
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                log.error(f"Strategy order error: {e}")

    def _aggressive_tick(self, books: dict[str, OrderBook]) -> None:
        """Place aggressive IOC orders on Group A products in parallel."""
        now = time.monotonic()
        futures = []

        for symbol in GROUP_A:
            # Backoff: skip symbol if too many recent misses
            if now < self._aggressive_backoff_until.get(symbol, 0):
                continue

            theo = self.theo_engine.get_theo(symbol)
            if theo is None:
                continue

            market_mid = self.price_tracker.get_mid(symbol)
            if market_mid is None:
                continue

            # Skip aggressive IOC when theo diverges >5% from market
            if market_mid > 0 and abs(theo - market_mid) / market_mid > 0.05:
                continue

            position = self.inventory.get_position(symbol)
            config = STRATEGY_CONFIGS.get(symbol)
            if config is None:
                continue

            book = books.get(symbol)
            sym_best_ask = best_ask(book) if book else None
            sym_best_bid = best_bid(book) if book else None

            result = compute_aggressive_ioc(
                symbol, market_mid, theo, position, config,
                threshold=AGGRESSIVE_THRESHOLD,
                best_ask=sym_best_ask,
                best_bid=sym_best_bid,
            )
            if result is None:
                continue

            side_str, price, size = result
            side = Side.BUY if side_str == "BUY" else Side.SELL

            log.info(
                f"AGGRESSIVE: {symbol} {side_str} {size}x @ {price} "
                f"(theo={theo:.0f}, market={market_mid:.0f}, "
                f"mispricing={(theo-market_mid)/theo*100:.2f}%)"
            )

            order = OrderRequest(symbol, price, side, size)
            futures.append(self._pool.submit(self._execute_aggressive_ioc, symbol, order, side_str, price, now))

        # Wait for all aggressive IOCs to complete
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                log.error(f"Aggressive IOC error: {e}")

    def _execute_aggressive_ioc(self, symbol: str, order: OrderRequest, side_str: str, price: float, now: float) -> None:
        """Execute a single aggressive IOC (runs in thread pool)."""
        resp = self.safe_send_ioc(order)
        if resp and resp.filled > 0:
            log.info(f"AGGRESSIVE FILL: {symbol} {side_str} {resp.filled}x @ {price}")
            self._aggressive_misses[symbol] = 0
            self._aggressive_backoff_until[symbol] = 0
        else:
            misses = self._aggressive_misses.get(symbol, 0) + 1
            self._aggressive_misses[symbol] = misses
            if misses >= 3:
                self._aggressive_backoff_until[symbol] = now + 30.0
                log.info(f"AGGRESSIVE BACKOFF: {symbol} after {misses} consecutive misses")

    def _reconcile_orders(
        self,
        symbol: str,
        bid_price: float | None,
        ask_price: float | None,
        bid_size: int,
        ask_size: int,
        need_bid: bool,
        need_ask: bool,
    ) -> None:
        """Cancel stale orders and place new ones. Each op costs 1 REST request."""
        # Cancel stale bid
        if need_bid:
            old_bid_id = self.resting_orders.get_order_id(symbol, "BUY")
            if old_bid_id is not None:
                try:
                    self.safe_cancel_order(old_bid_id)
                except Exception:
                    pass
                self.resting_orders.clear_order(symbol, "BUY")

            # Place new bid
            if bid_price is not None and bid_size > 0:
                order = OrderRequest(symbol, bid_price, Side.BUY, bid_size)
                resp = self.safe_send_order(order)
                if resp:
                    self.resting_orders.record_order(symbol, "BUY", resp.id, bid_price)
                    if resp.filled > 0:
                        log.info(
                            f"STRAT BID FILL: {symbol} BUY {resp.filled}x @ {bid_price}"
                        )

        # Cancel stale ask
        if need_ask:
            old_ask_id = self.resting_orders.get_order_id(symbol, "SELL")
            if old_ask_id is not None:
                try:
                    self.safe_cancel_order(old_ask_id)
                except Exception:
                    pass
                self.resting_orders.clear_order(symbol, "SELL")

            # Place new ask
            if ask_price is not None and ask_size > 0:
                order = OrderRequest(symbol, ask_price, Side.SELL, ask_size)
                resp = self.safe_send_order(order)
                if resp:
                    self.resting_orders.record_order(symbol, "SELL", resp.id, ask_price)
                    if resp.filled > 0:
                        log.info(
                            f"STRAT ASK FILL: {symbol} SELL {resp.filled}x @ {ask_price}"
                        )

    # --- PnL Logging ---

    def _log_pnl(self) -> None:
        """Fetch and log PnL + positions + theos from the exchange."""
        try:
            pnl = self.safe_get_pnl()
            positions = self.inventory.positions
            theos = self.theo_engine.get_all_theos()
            log.info(f"=== PnL: {pnl} | Positions: {dict(positions)} ===")
            log.info(f"=== Theos: {theos} ===")
        except Exception as e:
            log.error(f"PnL fetch failed: {e}")

    # --- Scratch & Sync ---

    def _execute_scratch(self, report) -> None:
        """Handle partial fills by reversing excess positions."""
        with self._books_lock:
            fresh_books = dict(self._books)

        scratch_plan = self.scratch.plan_scratch(report, fresh_books)
        if scratch_plan is None:
            return

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

        scratch_report = self.executor.execute_ioc_batch(scratch_plan.orders)

        # DON'T apply_fills here — on_trades SSE handles it (avoids double-counting)
        self.risk.update_positions(self.inventory.positions)

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
