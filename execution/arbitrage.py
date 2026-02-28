"""ETF Arbitrage Engine.

Monitors LON_ETF vs (TIDE_SPOT + WX_SPOT + LHR_COUNT).
When a mispricing exceeds the threshold, executes the 4-leg arb.
"""
import logging

from bot_template import OrderBook, OrderRequest, Side
from config import MAX_POSITION
from utils.helpers import best_bid, best_ask

log = logging.getLogger("arb")

COMPONENT_SYMBOLS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ETF_SYMBOL = "LON_ETF"


class ArbitrageEngine:
    def __init__(self, min_edge: float = 2.0):
        """
        Args:
            min_edge: Minimum price discrepancy to trigger arb.
                      Must exceed expected slippage + tick costs.
                      Start conservatively at 2-3 ticks.
        """
        self.min_edge = min_edge

    def check_and_generate_orders(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
        max_volume: int = 5,
    ) -> list[OrderRequest]:
        """Check for arb opportunity and return orders to execute.

        Returns empty list if no opportunity or if position limits
        would be breached.
        """
        etf_book = books.get(ETF_SYMBOL)
        component_books = {s: books.get(s) for s in COMPONENT_SYMBOLS}

        if etf_book is None or any(b is None for b in component_books.values()):
            return []

        # Get tradeable prices (excluding own orders)
        etf_bid = best_bid(etf_book)
        etf_ask = best_ask(etf_book)
        comp_bids = {s: best_bid(b) for s, b in component_books.items()}
        comp_asks = {s: best_ask(b) for s, b in component_books.items()}

        if etf_bid is None or etf_ask is None:
            return []
        if any(v is None for v in comp_bids.values()) or any(v is None for v in comp_asks.values()):
            return []

        # Case 1: ETF overpriced -> sell ETF, buy components
        comp_ask_sum = sum(comp_asks.values())
        edge_sell_etf = etf_bid - comp_ask_sum
        if edge_sell_etf >= self.min_edge:
            log.info(f"ARB: Sell ETF @ {etf_bid}, Buy components @ {comp_ask_sum}, edge={edge_sell_etf:.1f}")
            vol = self._safe_volume(max_volume, ETF_SYMBOL, Side.SELL, COMPONENT_SYMBOLS, Side.BUY, positions)
            if vol > 0:
                orders = [OrderRequest(ETF_SYMBOL, etf_bid, Side.SELL, vol)]
                for s in COMPONENT_SYMBOLS:
                    orders.append(OrderRequest(s, comp_asks[s], Side.BUY, vol))
                return orders

        # Case 2: ETF underpriced -> buy ETF, sell components
        comp_bid_sum = sum(comp_bids.values())
        edge_buy_etf = comp_bid_sum - etf_ask
        if edge_buy_etf >= self.min_edge:
            log.info(f"ARB: Buy ETF @ {etf_ask}, Sell components @ {comp_bid_sum}, edge={edge_buy_etf:.1f}")
            vol = self._safe_volume(max_volume, ETF_SYMBOL, Side.BUY, COMPONENT_SYMBOLS, Side.SELL, positions)
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
    ) -> int:
        """Compute max volume respecting +/-100 position limits on all legs."""
        vol = desired
        # Check ETF leg
        etf_pos = positions.get(etf_symbol, 0)
        if etf_side == Side.BUY:
            vol = min(vol, MAX_POSITION - etf_pos)
        else:
            vol = min(vol, MAX_POSITION + etf_pos)
        # Check component legs
        for s in comp_symbols:
            pos = positions.get(s, 0)
            if comp_side == Side.BUY:
                vol = min(vol, MAX_POSITION - pos)
            else:
                vol = min(vol, MAX_POSITION + pos)
        return max(0, vol)
