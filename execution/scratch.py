"""Scratch routine for flattening partial arb fills.

After concurrent execution, if legs filled unequally, this module
generates aggressive reverse IOC orders to close the excess exposure,
taking a small certain loss to prevent a large directional loss.
"""

import logging
from dataclasses import dataclass

from bot_template import OrderBook, OrderRequest, Side
from execution.executor import ExecutionReport
from utils.helpers import best_bid, best_ask, snap_to_tick

log = logging.getLogger("scratch")

SCRATCH_AGGRESSION_TICKS = 2


@dataclass
class ScratchPlan:
    """Plan for reversing excess fills."""

    orders: list[OrderRequest]
    expected_cost: float  # estimated P&L impact (positive = loss)
    excess_by_leg: dict[str, int]  # product -> excess volume to reverse


class ScratchRoutine:
    """Detects partial fills and generates reverse orders to flatten excess.

    After ConcurrentExecutor returns an ExecutionReport, call
    needs_scratch() then plan_scratch(). Execute the plan through
    the same ConcurrentExecutor.
    """

    def __init__(self, aggression_ticks: int = SCRATCH_AGGRESSION_TICKS):
        self.aggression_ticks = aggression_ticks

    def needs_scratch(self, report: ExecutionReport) -> bool:
        """True if legs filled unequally and scratch is needed."""
        return not report.all_filled_equal

    def plan_scratch(
        self,
        report: ExecutionReport,
        books: dict[str, OrderBook],
    ) -> ScratchPlan | None:
        """Generate reverse orders to flatten excess fills.

        The matched/hedged portion is min(filled) across all legs.
        Any leg that filled more than the minimum gets a reverse
        IOC order for the excess.

        Args:
            report: ExecutionReport from ConcurrentExecutor.
            books: Current order book snapshots (from SSE cache).

        Returns:
            ScratchPlan with orders, or None if no scratch needed.
        """
        if report.all_filled_equal:
            return None

        target = report.min_filled
        orders: list[OrderRequest] = []
        total_cost = 0.0
        excess_by_leg: dict[str, int] = {}

        for lr in report.legs:
            excess = lr.filled - target
            if excess <= 0:
                continue

            product = lr.order.product
            orig_side = lr.order.side
            orig_price = lr.order.price
            reverse_side = Side.SELL if orig_side == Side.BUY else Side.BUY

            # Aggressive price to guarantee fill
            book = books.get(product)
            price = self._aggressive_price(book, reverse_side, orig_price)

            orders.append(
                OrderRequest(
                    product=product,
                    price=price,
                    side=reverse_side,
                    volume=excess,
                )
            )
            excess_by_leg[product] = excess

            # Estimate cost (positive = loss)
            if orig_side == Side.BUY:
                # Bought at orig_price, now selling at (lower) price
                leg_cost = (orig_price - price) * excess
            else:
                # Sold at orig_price, now buying at (higher) price
                leg_cost = (price - orig_price) * excess
            total_cost += leg_cost

        if not orders:
            return None

        plan = ScratchPlan(
            orders=orders,
            expected_cost=total_cost,
            excess_by_leg=excess_by_leg,
        )

        log.warning(
            f"SCRATCH PLAN: matched_fill={target}, "
            f"excess={excess_by_leg}, "
            f"estimated_cost={total_cost:.2f}"
        )
        return plan

    def _aggressive_price(
        self,
        book: OrderBook | None,
        side: Side,
        fallback_price: float,
    ) -> float:
        """Compute an aggressive price that should guarantee a fill.

        For a scratch SELL: best_bid - aggression_ticks * tick_size
        For a scratch BUY:  best_ask + aggression_ticks * tick_size
        Falls back to the original order price if no book data.
        """
        if book is None:
            return fallback_price

        tick = book.tick_size

        if side == Side.SELL:
            bid = best_bid(book)
            if bid is not None:
                aggressive = bid - self.aggression_ticks * tick
                return snap_to_tick(aggressive, tick, direction="down")
            return fallback_price
        else:  # BUY
            ask = best_ask(book)
            if ask is not None:
                aggressive = ask + self.aggression_ticks * tick
                return snap_to_tick(aggressive, tick, direction="up")
            return fallback_price
