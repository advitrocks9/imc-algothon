"""Main trading bot."""
import threading
import time
import logging

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Trade, Side
from config import EXCHANGE_URL, USERNAME, PASSWORD, SYMBOLS
from execution.arbitrage import ArbitrageEngine
from risk.manager import RiskManager
from utils.rate_limiter import RateLimiter
from utils.helpers import best_bid, best_ask, mid_price, snap_to_tick

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot")


class TradingBot(BaseBot):
    def __init__(self):
        super().__init__(EXCHANGE_URL, USERNAME, PASSWORD)
        self.risk = RiskManager()
        self.limiter = RateLimiter(max_rps=1.0)
        self.arb_engine = ArbitrageEngine(min_edge=2.0)

        # Latest order book snapshots (updated by SSE)
        self._books_lock = threading.Lock()
        self._books: dict[str, OrderBook] = {}

        # Theos (updated by theo engine)
        self._theos_lock = threading.Lock()
        self._theos: dict[str, float | None] = {s: None for s in SYMBOLS}

        # Confidence intervals (half-width)
        self._confidence: dict[str, float] = {s: 999.0 for s in SYMBOLS}

    # --- SSE Callbacks ---

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Cache latest book snapshot. Do NOT place orders here."""
        with self._books_lock:
            self._books[orderbook.product] = orderbook

    def on_trades(self, trade: Trade) -> None:
        side = "BOUGHT" if trade.buyer == self.username else "SOLD"
        log.info(f"FILL: {side} {trade.volume}x {trade.product} @ {trade.price}")

    # --- Theo Access ---

    def get_theo(self, symbol: str) -> float | None:
        with self._theos_lock:
            return self._theos.get(symbol)

    def set_theo(self, symbol: str, value: float, confidence: float = 999.0) -> None:
        with self._theos_lock:
            self._theos[symbol] = value
            self._confidence[symbol] = confidence

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

        # Initial position sync
        self.risk.update_positions(self.safe_get_positions())
        log.info(f"Positions: {self.risk.positions}")

        # List products
        self.limiter.acquire()
        products = {p.symbol: p for p in self.get_products()}
        log.info(f"Products: {list(products.keys())}")

        try:
            while True:
                self._main_tick(products)
        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self.stop()

    def _main_tick(self, products: dict) -> None:
        """One iteration of the main trading loop."""
        # 1. Read cached SSE books (no REST call)
        with self._books_lock:
            books = dict(self._books)

        # 2. Check ETF arbitrage
        arb_orders = self.arb_engine.check_and_generate_orders(
            books, self.risk.positions, max_volume=5
        )
        if arb_orders:
            for order in arb_orders:
                resp = self.safe_send_ioc(order)
                if resp:
                    log.info(f"ARB executed: {order.product} {order.side} {resp.filled} filled @ {order.price}")

        # 3. Sync positions
        self.risk.update_positions(self.safe_get_positions())

        time.sleep(2)
