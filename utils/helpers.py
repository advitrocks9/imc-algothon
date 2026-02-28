"""Shared helper functions."""
import math

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side


def send_ioc(bot: BaseBot, order: OrderRequest) -> OrderResponse | None:
    """Simulate IOC: place order, immediately cancel unfilled remainder."""
    resp = bot.send_order(order)
    if resp and resp.volume > 0:
        bot.cancel_order(resp.id)
    return resp


def best_bid(ob: OrderBook) -> float | None:
    """Best bid price excluding own orders."""
    for o in ob.buy_orders:
        if o.volume - o.own_volume > 0:
            return o.price
    return None


def best_ask(ob: OrderBook) -> float | None:
    """Best ask price excluding own orders."""
    for o in ob.sell_orders:
        if o.volume - o.own_volume > 0:
            return o.price
    return None


def mid_price(ob: OrderBook) -> float | None:
    b = best_bid(ob)
    a = best_ask(ob)
    if b is not None and a is not None:
        return (b + a) / 2.0
    return None


def snap_to_tick(price: float, tick: float, direction: str = "nearest") -> float:
    """Snap a price to the nearest valid tick. Direction: 'up', 'down', 'nearest'."""
    if direction == "down":
        return math.floor(price / tick) * tick
    elif direction == "up":
        return math.ceil(price / tick) * tick
    else:
        return round(price / tick) * tick
