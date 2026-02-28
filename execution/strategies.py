"""Per-market strategy configurations and order computation.

Strategy types:
  - mean_reversion: Group A products, quote both sides around slow EMA
  - etf_synthetic:  LON_ETF, quote around sum-of-component mids
  - trend_follow:   Groups B-D, directional only on EMA crossover
  - derivative_arb: LON_FLY, quote around computed theo from LON_ETF
"""
from config import MAX_POSITION
from utils.helpers import snap_to_tick

# --- Strategy configs per symbol ---

STRATEGY_CONFIGS = {
    # GROUP A: mean-reversion, quote both sides, aggressive sizing
    "TIDE_SPOT": {
        "type": "mean_reversion",
        "entry_threshold": 0.15,
        "exit_threshold": 0.05,
        "max_position": 30,
        "order_size": 5,
        "spread_ticks": 3,
        "quote_both_sides": True,
    },
    "WX_SPOT": {
        "type": "mean_reversion",
        "entry_threshold": 0.15,
        "exit_threshold": 0.05,
        "max_position": 30,
        "order_size": 5,
        "spread_ticks": 4,
        "quote_both_sides": True,
    },
    "LHR_COUNT": {
        "type": "mean_reversion",
        "entry_threshold": 0.15,
        "exit_threshold": 0.05,
        "max_position": 25,
        "order_size": 5,
        "spread_ticks": 5,
        "quote_both_sides": True,
    },
    "LON_ETF": {
        "type": "etf_synthetic",
        "entry_threshold": 0.10,
        "exit_threshold": 0.05,
        "max_position": 30,
        "order_size": 5,
        "spread_ticks": 5,
        "quote_both_sides": True,
    },
    # GROUP B-D: trend-following, directional only, conservative
    "TIDE_SWING": {
        "type": "trend_follow",
        "entry_threshold": 0.20,
        "exit_threshold": 0.08,
        "max_position": 15,
        "order_size": 3,
        "spread_ticks": 5,
        "quote_both_sides": False,
    },
    "WX_SUM": {
        "type": "trend_follow",
        "entry_threshold": 0.20,
        "exit_threshold": 0.08,
        "max_position": 15,
        "order_size": 3,
        "spread_ticks": 5,
        "quote_both_sides": False,
    },
    "LHR_INDEX": {
        "type": "trend_follow",
        "entry_threshold": 0.20,
        "exit_threshold": 0.08,
        "max_position": 15,
        "order_size": 3,
        "spread_ticks": 5,
        "quote_both_sides": False,
    },
    # GROUP E: derivative, quote around LON_FLY theo
    "LON_FLY": {
        "type": "derivative_arb",
        "entry_threshold": 0.25,
        "exit_threshold": 0.10,
        "max_position": 15,
        "order_size": 3,
        "spread_ticks": 8,
        "quote_both_sides": True,
    },
}


# --- LON_FLY payoff ---

def lon_fly_payoff(s: float) -> float:
    """Compute LON_FLY settlement given LON_ETF settlement value s."""
    return (
        2.0 * max(0.0, 6200.0 - s)
        + 1.0 * max(0.0, s - 6200.0)
        - 2.0 * max(0.0, s - 6600.0)
        + 3.0 * max(0.0, s - 7000.0)
    )


# --- ETF synthetic mid ---

def compute_etf_synthetic_mid(price_tracker) -> float | None:
    """Sum of component mid prices for LON_ETF fair value."""
    tide = price_tracker.get_mid("TIDE_SPOT")
    wx = price_tracker.get_mid("WX_SPOT")
    lhr = price_tracker.get_mid("LHR_COUNT")
    if tide is not None and wx is not None and lhr is not None:
        return tide + wx + lhr
    return None


# --- Position sizing ---

def compute_order_size(
    signal: float,
    current_position: int,
    config: dict,
) -> tuple[int, int]:
    """Returns (bid_size, ask_size) respecting position limits and signal.

    Asymmetric: stronger signal → larger size in signal direction.
    Inventory scaling: reduce size as position approaches max.
    """
    base = config["order_size"]
    max_pos = config["max_position"]

    buy_headroom = max(0, MAX_POSITION - current_position)
    sell_headroom = max(0, MAX_POSITION + current_position)

    # Scale down as position approaches strategy max
    pos_ratio = abs(current_position) / max_pos if max_pos > 0 else 1.0
    inv_scale = max(0.2, 1.0 - pos_ratio)

    if signal > 0:
        bid_size = int(base * (1 + abs(signal)) * inv_scale)
        ask_size = int(base * (1 - abs(signal) * 0.5) * inv_scale)
    elif signal < 0:
        bid_size = int(base * (1 - abs(signal) * 0.5) * inv_scale)
        ask_size = int(base * (1 + abs(signal)) * inv_scale)
    else:
        bid_size = int(base * inv_scale)
        ask_size = int(base * inv_scale)

    bid_size = min(max(1, bid_size), buy_headroom)
    ask_size = min(max(1, ask_size), sell_headroom)

    # Hard cap: stop building position beyond strategy max
    if current_position >= max_pos:
        bid_size = 0
    if current_position <= -max_pos:
        ask_size = 0

    return (bid_size, ask_size)


# --- Quote pricing ---

def compute_quote_prices(
    mid: float,
    signal: float,
    position: int,
    config: dict,
    tick_size: float = 1.0,
) -> tuple[float, float]:
    """Compute bid/ask with inventory skew and signal skew.

    Inventory skew: when long, lower both quotes (more aggressive sell).
    Signal skew: shift in signal direction.
    """
    half_spread = config["spread_ticks"] * tick_size / 2.0
    max_pos = config["max_position"]

    # Inventory skew: push quotes away from position direction
    skew = 0.0
    if max_pos > 0:
        skew = -(position / max_pos) * half_spread * 0.5

    # Signal skew
    signal_skew = signal * half_spread * 0.3

    bid_price = snap_to_tick(mid - half_spread + skew + signal_skew, tick_size, "down")
    ask_price = snap_to_tick(mid + half_spread + skew + signal_skew, tick_size, "up")

    # Sanity: bid must be strictly below ask
    if bid_price >= ask_price:
        bid_price = ask_price - tick_size

    return (bid_price, ask_price)


# --- Top-level: compute desired orders for a symbol ---

def compute_desired_orders(
    symbol: str,
    price_tracker,
    position: int,
) -> tuple[float | None, float | None, int, int]:
    """Returns (bid_price, ask_price, bid_size, ask_size) or Nones if no trade.

    Handles strategy type dispatch (mean_reversion, etf_synthetic,
    trend_follow, derivative_arb).
    """
    config = STRATEGY_CONFIGS.get(symbol)
    if config is None:
        return (None, None, 0, 0)

    signal = price_tracker.get_signal(symbol)
    stype = config["type"]

    # Determine fair value (mid) based on strategy type
    if stype == "etf_synthetic":
        mid = compute_etf_synthetic_mid(price_tracker)
    elif stype == "derivative_arb":
        etf_mid = price_tracker.get_mid("LON_ETF")
        if etf_mid is None:
            etf_mid = compute_etf_synthetic_mid(price_tracker)
        mid = lon_fly_payoff(etf_mid) if etf_mid is not None else None
    else:
        mid = price_tracker.get_mid(symbol)

    if mid is None:
        return (None, None, 0, 0)

    # For trend_follow: only quote one side (directional)
    if stype == "trend_follow":
        if abs(signal) < config["entry_threshold"]:
            # Signal too weak — don't trade
            if abs(position) > 0 and abs(signal) < config["exit_threshold"]:
                # Flatten: quote the reducing side aggressively
                return _flatten_orders(mid, position, config)
            return (None, None, 0, 0)
        # Strong signal — quote only the directional side
        return _directional_orders(mid, signal, position, config)

    # For mean_reversion / etf_synthetic / derivative_arb: quote both sides
    bid_price, ask_price = compute_quote_prices(mid, signal, position, config)
    bid_size, ask_size = compute_order_size(signal, position, config)

    if not config.get("quote_both_sides", True):
        if signal > 0:
            ask_price, ask_size = None, 0
        elif signal < 0:
            bid_price, bid_size = None, 0

    return (bid_price, ask_price, bid_size, ask_size)


def _directional_orders(
    mid: float, signal: float, position: int, config: dict,
) -> tuple[float | None, float | None, int, int]:
    """Directional: only place order in signal direction."""
    half_spread = config["spread_ticks"] / 2.0
    base_size = config["order_size"]
    max_pos = config["max_position"]

    buy_headroom = max(0, MAX_POSITION - position)
    sell_headroom = max(0, MAX_POSITION + position)

    if signal > 0 and position < max_pos:
        bid_price = snap_to_tick(mid - half_spread, 1.0, "down")
        bid_size = min(base_size, buy_headroom)
        return (bid_price, None, bid_size, 0)
    elif signal < 0 and position > -max_pos:
        ask_price = snap_to_tick(mid + half_spread, 1.0, "up")
        ask_size = min(base_size, sell_headroom)
        return (None, ask_price, 0, ask_size)

    return (None, None, 0, 0)


def _flatten_orders(
    mid: float, position: int, config: dict,
) -> tuple[float | None, float | None, int, int]:
    """Place aggressive order to flatten position."""
    if position > 0:
        # Need to sell to flatten — aggressive ask (close to mid)
        ask_price = snap_to_tick(mid - 1, 1.0, "down")
        ask_size = min(abs(position), MAX_POSITION + position)
        return (None, ask_price, 0, min(ask_size, config["order_size"]))
    elif position < 0:
        # Need to buy to flatten — aggressive bid (close to mid)
        bid_price = snap_to_tick(mid + 1, 1.0, "up")
        bid_size = min(abs(position), MAX_POSITION - position)
        return (bid_price, None, min(bid_size, config["order_size"]), 0)
    return (None, None, 0, 0)
