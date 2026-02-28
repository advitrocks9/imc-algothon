"""ETF Arbitrage Engine.

Monitors LON_ETF vs (TIDE_SPOT + WX_SPOT + LHR_COUNT).
When a mispricing exceeds the threshold, executes the 4-leg arb.

Volume is clamped to the "weakest link" — the minimum of:
  - base target quantity
  - position limit headroom on all 4 legs
  - available book depth at the targeted price for all 4 legs
"""
import logging

from bot_template import OrderBook, OrderRequest, Side
from config import MAX_POSITION, MIN_BOOK_DEPTH
from utils.helpers import best_bid_with_volume, best_ask_with_volume

log = logging.getLogger("arb")

COMPONENT_SYMBOLS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ETF_SYMBOL = "LON_ETF"


class ArbitrageEngine:
    def __init__(self, min_edge: float = 2.0):
        """
        Args:
            min_edge: Minimum price discrepancy to trigger arb.
                      Must exceed expected slippage + scratch costs.
        """
        self.min_edge = min_edge

    def check_and_generate_orders(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
        max_volume: int = 5,
    ) -> list[OrderRequest]:
        """Check for arb opportunity and return orders to execute.

        Returns empty list if no opportunity, position limits breached,
        or any book leg has zero available volume.
        """
        etf_book = books.get(ETF_SYMBOL)
        component_books = {s: books.get(s) for s in COMPONENT_SYMBOLS}

        if etf_book is None or any(b is None for b in component_books.values()):
            return []

        # Get tradeable prices AND available volumes (excluding own orders)
        etf_bid_info = best_bid_with_volume(etf_book)
        etf_ask_info = best_ask_with_volume(etf_book)
        comp_bid_infos = {s: best_bid_with_volume(b) for s, b in component_books.items()}
        comp_ask_infos = {s: best_ask_with_volume(b) for s, b in component_books.items()}

        if etf_bid_info is None or etf_ask_info is None:
            return []
        if any(v is None for v in comp_bid_infos.values()):
            return []
        if any(v is None for v in comp_ask_infos.values()):
            return []

        # Unpack prices and volumes
        etf_bid, etf_bid_vol = etf_bid_info
        etf_ask, etf_ask_vol = etf_ask_info
        comp_asks = {s: info[0] for s, info in comp_ask_infos.items()}
        comp_ask_vols = {s: info[1] for s, info in comp_ask_infos.items()}
        comp_bids = {s: info[0] for s, info in comp_bid_infos.items()}
        comp_bid_vols = {s: info[1] for s, info in comp_bid_infos.items()}

        # Case 1: ETF overpriced -> sell ETF (hit bid), buy components (lift asks)
        comp_ask_sum = sum(comp_asks.values())
        edge_sell_etf = etf_bid - comp_ask_sum
        if edge_sell_etf >= self.min_edge:
            book_vols = {ETF_SYMBOL: etf_bid_vol}
            book_vols.update(comp_ask_vols)

            # Skip if any leg has insufficient book depth
            if any(v < MIN_BOOK_DEPTH for v in book_vols.values()):
                thin = {s: v for s, v in book_vols.items() if v < MIN_BOOK_DEPTH}
                log.debug(f"ARB: skipping sell-ETF — thin books: {thin}")
                return []

            vol, bottleneck = self._safe_volume(
                max_volume, ETF_SYMBOL, Side.SELL,
                COMPONENT_SYMBOLS, Side.BUY, positions, book_vols,
            )
            log.info(
                f"ARB: Sell ETF @ {etf_bid}, Buy components @ {comp_ask_sum}, "
                f"edge={edge_sell_etf:.1f}, clamped_vol={vol}, "
                f"bottleneck={bottleneck}, "
                f"book_depths={book_vols}"
            )
            if vol > 0:
                orders = [OrderRequest(ETF_SYMBOL, etf_bid, Side.SELL, vol)]
                for s in COMPONENT_SYMBOLS:
                    orders.append(OrderRequest(s, comp_asks[s], Side.BUY, vol))
                return orders

        # Case 2: ETF underpriced -> buy ETF (lift ask), sell components (hit bids)
        comp_bid_sum = sum(comp_bids.values())
        edge_buy_etf = comp_bid_sum - etf_ask
        if edge_buy_etf >= self.min_edge:
            book_vols = {ETF_SYMBOL: etf_ask_vol}
            book_vols.update(comp_bid_vols)

            # Skip if any leg has insufficient book depth
            if any(v < MIN_BOOK_DEPTH for v in book_vols.values()):
                thin = {s: v for s, v in book_vols.items() if v < MIN_BOOK_DEPTH}
                log.debug(f"ARB: skipping buy-ETF — thin books: {thin}")
                return []

            vol, bottleneck = self._safe_volume(
                max_volume, ETF_SYMBOL, Side.BUY,
                COMPONENT_SYMBOLS, Side.SELL, positions, book_vols,
            )
            log.info(
                f"ARB: Buy ETF @ {etf_ask}, Sell components @ {comp_bid_sum}, "
                f"edge={edge_buy_etf:.1f}, clamped_vol={vol}, "
                f"bottleneck={bottleneck}, "
                f"book_depths={book_vols}"
            )
            if vol > 0:
                orders = [OrderRequest(ETF_SYMBOL, etf_ask, Side.BUY, vol)]
                for s in COMPONENT_SYMBOLS:
                    orders.append(OrderRequest(s, comp_bids[s], Side.SELL, vol))
                return orders

        return []

    def _safe_volume(
        self,
        desired: int,
        etf_symbol: str,
        etf_side: Side,
        comp_symbols: list[str],
        comp_side: Side,
        positions: dict[str, int],
        book_volumes: dict[str, int],
    ) -> tuple[int, str | None]:
        """Compute max volume respecting position limits AND book depth.

        Returns (clamped_volume, bottleneck_symbol). bottleneck_symbol is
        the product that caused the tightest constraint, or None if the
        base desired volume was the binding constraint.
        """
        vol = desired
        bottleneck: str | None = None

        # --- Position limit clamp ---
        etf_pos = positions.get(etf_symbol, 0)
        if etf_side == Side.BUY:
            limit = MAX_POSITION - etf_pos
        else:
            limit = MAX_POSITION + etf_pos
        if limit < vol:
            vol = limit
            bottleneck = f"{etf_symbol}(pos)"

        for s in comp_symbols:
            pos = positions.get(s, 0)
            if comp_side == Side.BUY:
                limit = MAX_POSITION - pos
            else:
                limit = MAX_POSITION + pos
            if limit < vol:
                vol = limit
                bottleneck = f"{s}(pos)"

        # --- Book depth clamp (the "weakest link") ---
        for symbol, available in book_volumes.items():
            if available < vol:
                vol = available
                bottleneck = f"{symbol}(book)"

        return (max(0, vol), bottleneck)
