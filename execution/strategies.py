"""Per-market strategy configurations and order computation.

Strategy uses theo-based fair values from TheoEngine when available,
falling back to market mid-prices. Signal is computed as the deviation
of market price from theo (mispricing signal), not EMA momentum.

Strategy types:
  - theo_mm:        Group A products, quote both sides around theo
  - etf_synthetic:  LON_ETF, quote around theo (= sum of component theos)
  - theo_accum:     Accumulator products (WX_SUM, TIDE_SWING), quote around theo
  - derivative_arb: LON_FLY, quote around computed theo from LON_ETF distribution
  - cautious:       LHR_INDEX, wide spread, low confidence
"""
from config import MAX_POSITION
from utils.helpers import snap_to_tick

# --- Strategy configs per symbol ---

STRATEGY_CONFIGS = {
    # GROUP A: theo-based market making, aggressive sizing
    "TIDE_SPOT": {
        "type": "theo_mm",
        "max_position": 30,
        "order_size": 3,
        "spread_ticks": 3,
        "quote_both_sides": True,
    },
    "WX_SPOT": {
        "type": "theo_mm",
        "max_position": 30,
        "order_size": 3,
        "spread_ticks": 3,
        "quote_both_sides": True,
    },
    "LHR_COUNT": {
        "type": "theo_mm",
        "max_position": 30,
        "order_size": 3,
        "spread_ticks": 4,
        "quote_both_sides": True,
        "theo_fade_factor": 0.5,
    },
    "LON_ETF": {
        "type": "etf_synthetic",
        "max_position": 30,
        "order_size": 3,
        "spread_ticks": 4,
        "quote_both_sides": True,
    },
    # GROUP B-C: accumulator products with theo
    "TIDE_SWING": {
        "type": "theo_accum",
        "max_position": 20,
        "order_size": 3,
        "spread_ticks": 3,
        "quote_both_sides": True,
    },
    "WX_SUM": {
        "type": "theo_accum",
        "max_position": 20,
        "order_size": 3,
        "spread_ticks": 3,
        "quote_both_sides": True,
    },
    # GROUP D: cautious, low confidence
    "LHR_INDEX": {
        "type": "cautious",
        "max_position": 8,
        "order_size": 3,
        "spread_ticks": 5,
        "quote_both_sides": True,
    },
    # GROUP E: derivative pricing from LON_ETF distribution
    "LON_FLY": {
        "type": "derivative_arb",
        "max_position": 20,
        "order_size": 3,
        "spread_ticks": 3,
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


# --- ETF synthetic mid (fallback when no theo) ---

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

    signal > 0 means theo > market (should buy more).
    signal < 0 means theo < market (should sell more).
    """
    base = config["order_size"]
    max_pos = config["max_position"]

    buy_headroom = max(0, MAX_POSITION - current_position)
    sell_headroom = max(0, MAX_POSITION + current_position)

    # Scale down building direction as position approaches strategy max,
    # but keep unwinding direction at full size to flatten faster
    pos_ratio = abs(current_position) / max_pos if max_pos > 0 else 1.0
    build_scale = max(0.2, 1.0 - pos_ratio)

    if signal > 0:
        # Theo says buy: larger bids, smaller asks
        bid_scale = build_scale if current_position >= 0 else 1.0
        ask_scale = 1.0 if current_position > 0 else build_scale
        bid_size = int(base * (1 + abs(signal)) * bid_scale)
        ask_size = int(base * max(0.2, 1 - abs(signal)) * ask_scale)
    elif signal < 0:
        # Theo says sell: smaller bids, larger asks
        bid_scale = 1.0 if current_position < 0 else build_scale
        ask_scale = build_scale if current_position <= 0 else 1.0
        bid_size = int(base * max(0.2, 1 - abs(signal)) * bid_scale)
        ask_size = int(base * (1 + abs(signal)) * ask_scale)
    else:
        bid_scale = 1.0 if current_position < 0 else build_scale
        ask_scale = 1.0 if current_position > 0 else build_scale
        bid_size = int(base * bid_scale)
        ask_size = int(base * ask_scale)

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

    mid is the fair value (theo when available, market mid otherwise).
    signal is mispricing: (theo - market_mid) / theo, clamped to [-1, 1].
    """
    half_spread = config["spread_ticks"] * tick_size / 2.0
    max_pos = config["max_position"]

    # Inventory skew: push quotes to reduce position (aggressive mean-reversion)
    skew = 0.0
    if max_pos > 0:
        skew = -(position / max_pos) * half_spread * 2.0

    # Signal skew: gentle lean toward theo direction
    signal_skew = signal * half_spread * 0.5

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
    theo_engine=None,
) -> tuple[float | None, float | None, int, int]:
    """Returns (bid_price, ask_price, bid_size, ask_size) or Nones if no trade.

    Uses theo_engine for fair value when available, falls back to market mid.
    Signal is computed as mispricing: (theo - market) / theo.
    """
    config = STRATEGY_CONFIGS.get(symbol)
    if config is None:
        return (None, None, 0, 0)

    stype = config["type"]
    market_mid = price_tracker.get_mid(symbol)

    # Determine fair value based on strategy type and theo availability
    theo = None
    if theo_engine is not None:
        theo = theo_engine.get_theo(symbol)

    if stype == "etf_synthetic":
        # Prefer theo, fall back to synthetic mid from market
        if theo is not None:
            mid = theo
        else:
            mid = compute_etf_synthetic_mid(price_tracker)
    elif stype == "derivative_arb":
        # LON_FLY: prefer theo, fall back to payoff from market LON_ETF
        if theo is not None:
            mid = theo
        else:
            etf_mid = price_tracker.get_mid("LON_ETF")
            if etf_mid is None:
                etf_mid = compute_etf_synthetic_mid(price_tracker)
            mid = lon_fly_payoff(etf_mid) if etf_mid is not None else None
    elif stype in ("theo_mm", "theo_accum"):
        # Prefer theo, fall back to market mid
        if theo is not None:
            mid = theo
        else:
            mid = market_mid
    elif stype == "cautious":
        # Low confidence: prefer market mid, use theo only for directional signal
        mid = market_mid if market_mid is not None else theo
    else:
        mid = market_mid

    if mid is None:
        return (None, None, 0, 0)

    # Theo-market divergence capping: when theo is >5% off market,
    # blend toward market to avoid massive directional bias
    if theo is not None and market_mid is not None and market_mid > 0:
        divergence = abs(theo - market_mid) / market_mid
        if divergence > 0.10:
            # Circuit breaker: theo is wildly off — stop quoting to avoid wrong-way fills
            return (None, None, 0, 0)
        if divergence > 0.05:
            mid = 0.3 * theo + 0.7 * market_mid

    # Inventory-driven theo fading for arb-priority products
    fade_factor = config.get("theo_fade_factor", 0)
    if fade_factor > 0:
        mid = mid - (position * fade_factor)

    # Compute signal: mispricing relative to theo
    if theo is not None and market_mid is not None and theo > 0:
        signal = max(-1.0, min(1.0, (theo - market_mid) / theo))
    else:
        # No theo — use zero signal (symmetric quoting)
        signal = 0.0

    # Quote both sides around fair value with signal-based skew
    bid_price, ask_price = compute_quote_prices(mid, signal, position, config)
    bid_size, ask_size = compute_order_size(signal, position, config)

    return (bid_price, ask_price, bid_size, ask_size)


def compute_aggressive_ioc(
    symbol: str,
    market_mid: float,
    theo: float,
    position: int,
    config: dict,
    threshold: float = 0.008,
    best_ask: float | None = None,
    best_bid: float | None = None,
) -> tuple[str, float, int] | None:
    """Compute an aggressive IOC order when market deviates from theo.

    Returns (side, price, size) or None if no aggressive order warranted.
    side is "BUY" or "SELL".
    Uses best_ask/best_bid for pricing when available (lifts the actual ask/hits the actual bid).
    """
    if theo <= 0 or market_mid <= 0:
        return None

    mispricing = (theo - market_mid) / theo

    if abs(mispricing) < threshold:
        return None

    max_pos = config["max_position"]
    base_size = config["order_size"]
    buy_headroom = max(0, MAX_POSITION - position)
    sell_headroom = max(0, MAX_POSITION + position)

    # Position-aware scaling: reduce aggressive size as position grows
    # in the signal direction. Hard stop at 60% of max_pos.
    hard_limit = int(max_pos * 0.6)

    if mispricing > 0 and position < max_pos:
        # Theo > market: buy aggressively (lift the ask)
        if position >= hard_limit:
            return None  # already long enough
        pos_scale = max(0.2, 1.0 - max(0, position) / hard_limit)
        if best_ask is not None:
            price = snap_to_tick(best_ask, 1.0, "up")
        else:
            price = snap_to_tick(market_mid + 1, 1.0, "up")
        size = min(max(1, int(base_size * pos_scale)), buy_headroom)
        if size > 0:
            return ("BUY", price, size)
    elif mispricing < 0 and position > -max_pos:
        # Theo < market: sell aggressively (hit the bid)
        if position <= -hard_limit:
            return None  # already short enough
        pos_scale = max(0.2, 1.0 - max(0, -position) / hard_limit)
        if best_bid is not None:
            price = snap_to_tick(best_bid, 1.0, "down")
        else:
            price = snap_to_tick(market_mid - 1, 1.0, "down")
        size = min(max(1, int(base_size * pos_scale)), sell_headroom)
        if size > 0:
            return ("SELL", price, size)

    return None
