# IMCity London Challenge — Complete Bot Specification

> **Purpose:** This document is a self-contained specification for a coding agent to build an algorithmic trading bot for the IMCity London Challenge. It includes every formula, API detail, data source, and code pattern needed. Follow the phases in order. Do not deviate from the architecture.

---

## Table of Contents

1. [Competition Context](#1-competition-context)
2. [Exchange API Reference](#2-exchange-api-reference)
3. [External Data Sources](#3-external-data-sources)
4. [Product Specifications & Settlement Formulae](#4-product-specifications--settlement-formulae)
5. [Scoring System](#5-scoring-system)
6. [Project Architecture](#6-project-architecture)
7. [Phase 0 — Boilerplate & Infrastructure](#7-phase-0--boilerplate--infrastructure)
8. [Phase 1 — ETF Arbitrage Engine](#8-phase-1--etf-arbitrage-engine)
9. [Phase 2 — Tidal, Weather & Airport Theo Models](#9-phase-2--tidal-weather--airport-theo-models)
10. [Phase 3 — Market Making on All Products](#10-phase-3--market-making-on-all-products)
11. [Phase 4 — Derivatives Pricing (TIDE_SWING, LHR_INDEX, LON_FLY)](#11-phase-4--derivatives-pricing-tide_swing-lhr_index-lon_fly)
12. [Operational Playbook](#12-operational-playbook)

---

## 1. Competition Context

### What It Is
A 24-hour market-making and trading competition on a proprietary central limit order book exchange. You trade 8 products whose settlement values are determined by real-time London data (Thames water level, weather, Heathrow flights).

### Timeline
- **Saturday 12:00 PM** — Competition starts, exchange opens (actually 2:30 PM for challenge exchange)
- **Sunday 12:00 PM** — All products settle against live data snapshots
- Challenge exchange closes at Sunday 12:30 PM

### Core Rules
- 1 account per team, 1 running bot per account
- No spoofing, layering, wash trading, or market manipulation
- Rate limit: **1 REST request per second** (SSE stream is exempt)
- Position limit: **±100 net contracts per product**
- All order types are GTC (Good-Til-Cancelled). Simulate IOC by placing + immediately cancelling.
- Cancel + re-place sends you to back of the price-time queue

### Matching Engine
Price-time priority. Incoming buys match the lowest resting sell first. Incoming sells match the highest resting buy first. At the same price, the earliest order wins.

---

## 2. Exchange API Reference

### 2.1 Bot Template (provided file: `bot_template.py`)

The file `bot_template.py` is the provided framework. **Do not modify this file.** Import from it. It provides:

```python
from bot_template import (
    BaseBot,        # Abstract base — subclass this
    OrderBook,      # Dataclass: .product, .tick_size, .buy_orders, .sell_orders
    Order,          # Dataclass: .price, .volume, .own_volume
    OrderRequest,   # Dataclass: product, price, side, volume
    OrderResponse,  # Dataclass: .id, .status, .volume, .filled, .side, .price, .product
    Trade,          # Dataclass: .timestamp, .product, .buyer, .seller, .volume, .price
    Side,           # Enum: Side.BUY, Side.SELL
    Product,        # Dataclass: .symbol, .tickSize, .startingPrice, .contractSize
)
```

Dependencies: `requests`, `sseclient-py`, `pandas`, `numpy`. Python 3.10+.

### 2.2 Connection

```python
TEST_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
CHALLENGE_URL = "REPLACE_WITH_CHALLENGE_URL"
```

Subclass `BaseBot(cmi_url, username, password)`. Call `.start()` to open SSE stream, `.stop()` to close.

### 2.3 SSE Callbacks

You **must** implement two callbacks:

```python
def on_orderbook(self, orderbook: OrderBook) -> None:
    # Fires on ANY orderbook change, INCLUDING your own orders/cancels/fills
    # ⚠️ CRITICAL: Guard against infinite loops. Never place orders inside
    # this callback unconditionally or you will be banned.

def on_trades(self, trade: Trade) -> None:
    # Fires on YOUR fills only
    # trade.buyer == self.username means you bought
    # trade.seller == self.username means you sold
```

### 2.4 REST Methods

All subject to 1 req/sec rate limit.

| Method | Returns | Notes |
|--------|---------|-------|
| `bot.get_products()` | `list[Product]` | `.symbol`, `.tickSize`, `.startingPrice`, `.contractSize` |
| `bot.get_orderbook(symbol)` | `OrderBook` | `.buy_orders` sorted best-first (highest), `.sell_orders` sorted best-first (lowest) |
| `bot.send_order(OrderRequest(...))` | `OrderResponse \| None` | GTC order. Returns None on failure. |
| `bot.send_orders([...])` | `list[OrderResponse]` | Sends orders concurrently via threads |
| `bot.get_orders(product=None)` | `list[dict]` | Your resting orders. Keys: `id`, `side`, `volume`, `price`, `product` |
| `bot.cancel_order(order_id)` | `None` | Cancel by ID |
| `bot.cancel_all_orders()` | `None` | Cancels all resting orders concurrently |
| `bot.get_positions()` | `dict[str, int]` | `{symbol: net_position}`. Positive = long, negative = short |
| `bot.get_pnl()` | `dict` | Current PnL breakdown |
| `bot.get_market_trades()` | `list[Trade]` | Incremental. Appends to `bot.trades`. Uses watermark. |

### 2.5 IOC Pattern

The exchange only supports GTC. Simulate Immediate-Or-Cancel:

```python
def send_ioc(bot: BaseBot, order: OrderRequest) -> OrderResponse | None:
    resp = bot.send_order(order)
    if resp and resp.volume > 0:  # unfilled remainder exists
        bot.cancel_order(resp.id)
    return resp
```

This consumes 2 requests (place + cancel). Budget accordingly with rate limits.

### 2.6 OrderBook Helper Patterns

```python
def get_best_bid(ob: OrderBook) -> float | None:
    """Best bid excluding own orders."""
    for o in ob.buy_orders:
        if o.volume - o.own_volume > 0:
            return o.price
    return None

def get_best_ask(ob: OrderBook) -> float | None:
    """Best ask excluding own orders."""
    for o in ob.sell_orders:
        if o.volume - o.own_volume > 0:
            return o.price
    return None

def get_mid(ob: OrderBook) -> float | None:
    bid = get_best_bid(ob)
    ask = get_best_ask(ob)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return None
```

### 2.7 Rate Limit Strategy

With 1 req/sec, a full cycle of cancel-all + quote 8 products = ~17 requests = ~17 seconds minimum. Design the main loop around this constraint:

- **Option A (Simple):** Single-threaded loop. Each iteration: cancel all → fetch positions → re-quote all products. Interval ~20s.
- **Option B (Staggered):** Cycle through products one at a time. Each tick: cancel + re-quote 1 product. Full cycle completes every ~16s but each product is refreshed every ~2s on average.
- **Option C (Event-driven):** Use SSE `on_orderbook` to track state. Only cancel/re-quote a product when its book has moved beyond a threshold. Minimises REST calls.

**Recommendation:** Start with Option A for simplicity. Upgrade to Option C once the basic system is working.

---

## 3. External Data Sources

### 3.1 Thames Tidal Level — EA Flood Monitoring API (free, no key)

```python
import requests
import pandas as pd

THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"

def get_thames(limit: int = 200) -> pd.DataFrame:
    """Fetch recent Thames tidal readings.

    Returns DataFrame with columns: time (tz-aware London), level (mAOD float).
    Level is in metres Above Ordnance Datum.
    - Multiply by 1000 for TIDE_SPOT (mm).
    - Compute 15-min diffs and multiply by 100 for TIDE_SWING (cm).
    """
    resp = requests.get(
        f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
        params={"_sorted": "", "_limit": limit},
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    df = pd.DataFrame(items)[["dateTime", "value"]].rename(
        columns={"dateTime": "time", "value": "level"}
    )
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
    return df.sort_values("time").reset_index(drop=True)
```

**Unit conversions:**
- Raw value is in **mAOD** (metres Above Ordnance Datum), e.g. `1.412`
- `TIDE_SPOT` settlement = `abs(level) × 1000` → e.g. `1412`
- `TIDE_SWING` uses 15-min diffs in **cm** → `abs(level_t - level_{t-1}) × 100`

### 3.2 London Weather — Open-Meteo (free, no key)

```python
LONDON_LAT, LONDON_LON = 51.5074, -0.1278

def get_weather(past_steps: int = 96, forecast_steps: int = 96) -> pd.DataFrame:
    """Fetch 15-min weather data: past + forecast.

    96 steps = 24 hours. Temperature returned in °C — MUST convert to °F.
    """
    variables = (
        "temperature_2m,apparent_temperature,relative_humidity_2m,"
        "precipitation,wind_speed_10m,cloud_cover,visibility"
    )
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LONDON_LAT,
            "longitude": LONDON_LON,
            "minutely_15": variables,
            "past_minutely_15": past_steps,
            "forecast_minutely_15": forecast_steps,
            "timezone": "Europe/London",
        },
    )
    resp.raise_for_status()
    m = resp.json()["minutely_15"]
    return pd.DataFrame({
        "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
        "temperature_c": m["temperature_2m"],
        "humidity": m["relative_humidity_2m"],
        "apparent_temperature": m["apparent_temperature"],
        "precipitation": m["precipitation"],
        "wind_speed": m["wind_speed_10m"],
        "cloud_cover": m["cloud_cover"],
        "visibility": m["visibility"],
    })

def c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0
```

**Critical:** Open-Meteo returns temperature in **°C**. All settlement formulae use **°F**. Always convert.

### 3.3 Heathrow Flights — AeroDataBox (RapidAPI, key required, ~150 free req/month)

```python
import json

AERODATABOX_KEY = "YOUR_API_KEY"  # Set via env var in production
AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"

def fetch_flights_range(
    airport: str = "LHR",
    from_local: str = "2026-02-28T12:00",
    to_local: str = "2026-02-29T00:00",
    filters: dict | None = None,
) -> dict:
    """Fetch flights in a time range (max 12h span).

    Returns {"arrivals": [...], "departures": [...]}.
    Must make 2 calls to cover full 24h (12h max per call).
    """
    params = "?direction=Both"
    if filters:
        for k, v in filters.items():
            params += f"&{k}={'true' if v else 'false'}"
    url = (
        f"https://{AERODATABOX_HOST}/flights/airports/iata/"
        f"{airport}/{from_local}/{to_local}{params}"
    )
    resp = requests.get(
        url,
        headers={
            "x-rapidapi-host": AERODATABOX_HOST,
            "x-rapidapi-key": AERODATABOX_KEY,
        },
    )
    resp.raise_for_status()
    return json.loads(resp.text)
```

**Budget:** ~150 free requests/month. Plan calls carefully. Fetch the full 24h window in 2 calls (2×12h) at startup, then re-fetch every ~2 hours to capture delays/cancellations. That's ~14 calls total.

---

## 4. Product Specifications & Settlement Formulae

All products have `tick_size = 1` and settlements ≥ 0.

### 4.1 TIDE_SPOT (Market 1 — Group A)

```
Settlement = abs(tidal_height_mAOD_at_noon_sunday) × 1000
```

Example: if water level is 1.412 mAOD → settlement = 1412.

### 4.2 TIDE_SWING (Market 2 — Group B)

```
Settlement = Σ strangle_value(diff_cm_i) for each 15-min interval

where diff_cm = abs(level_t - level_{t-15min}) × 100   (convert m → cm)

strangle_value(d) = max(0, 20 - d) + max(0, d - 25)
```

This is a **Put at strike 20cm / Call at strike 25cm** strangle on each 15-minute absolute difference.

**Validation examples:**
- diff = 9 cm → `max(0, 20-9) + max(0, 9-25) = 11 + 0 = 11` ✓
- diff = 33 cm → `max(0, 20-33) + max(0, 33-25) = 0 + 8 = 8` ✓
- diff = 22 cm → `max(0, 20-22) + max(0, 22-25) = 0 + 0 = 0` (inside the dead zone)

There are **96 intervals** in 24 hours (one every 15 minutes).

### 4.3 WX_SPOT (Market 3 — Group A)

```
Settlement = round(temperature_F × humidity_pct) at Sunday 12:00 PM
```

Example: 45.3°F × 82% → round(3714.6) = 3715. Note the `round()` — creates discrete possible settlements.

### 4.4 WX_SUM (Market 4 — Group C)

```
Settlement = Σ (temperature_F × humidity_pct / 100) for each 15-min interval
```

96 intervals over 24h. Example ballpark: 96 × (45 × 80 / 100) ≈ 3456.

### 4.5 LHR_COUNT (Market 5 — Group A)

```
Settlement = total_arrivals + total_departures over 24 hours
```

Typical Heathrow weekend: ~650–750 total movements.

### 4.6 LHR_INDEX (Market 6 — Group D)

```
Settlement = abs(Σ [100 × (arrivals_i - departures_i) / max(arrivals_i + departures_i, 1)])

where i = each 30-minute interval (48 intervals in 24h)
```

Note the final **absolute value** wrapping the entire sum.

### 4.7 LON_ETF (Market 7 — Group A)

```
Settlement = TIDE_SPOT + WX_SPOT + LHR_COUNT
```

Always ≥ 0 since all components are non-negative. The absolute value in the spec is redundant but technically present.

### 4.8 LON_FLY (Market 8 — Group E)

Options portfolio on `LON_ETF` settlement value `S`:

| Leg | Position |
|-----|----------|
| Long 2× | 6200 Strike Put |
| Long 1× | 6200 Strike Call |
| Short 2× | 6600 Strike Call |
| Long 3× | 7000 Strike Call |

**Piecewise payoff function:**

```python
def lon_fly_payoff(s: float) -> float:
    """Compute LON_FLY settlement given LON_ETF settlement value s."""
    return (
        2.0 * max(0.0, 6200.0 - s)    # long 2× 6200 put
        + 1.0 * max(0.0, s - 6200.0)  # long 1× 6200 call
        - 2.0 * max(0.0, s - 6600.0)  # short 2× 6600 call
        + 3.0 * max(0.0, s - 7000.0)  # long 3× 7000 call
    )
```

**Regime breakdown (for intuition and validation):**

| Regime | Condition | Simplified Payoff |
|--------|-----------|-------------------|
| 1 | S < 6200 | `12400 - 2S` |
| 2 | 6200 ≤ S < 6600 | `S - 6200` |
| 3 | 6600 ≤ S < 7000 | `7000 - S` |
| 4 | S ≥ 7000 | `2S - 14000` |

**Validation:**
- S=6000 → `12400 - 12000 = 400` ✓ and `2×200 + 0 - 0 + 0 = 400` ✓
- S=6400 → `6400 - 6200 = 200` ✓ and `0 + 200 - 0 + 0 = 200` ✓
- S=6800 → `7000 - 6800 = 200` ✓ and `0 + 600 - 2×200 + 0 = 200` ✓
- S=7000 → `7000 - 7000 = 0` ✓ (minimum of the V-shape)
- S=7200 → `2×7200 - 14000 = 400` ✓ and `0 + 1000 - 2×600 + 3×200 = 400` ✓

---

## 5. Scoring System

### 5.1 Product Groups

| Group | Markets | Weight |
|-------|---------|--------|
| A | TIDE_SPOT, WX_SPOT, LHR_COUNT, LON_ETF | **4×** |
| B | TIDE_SWING | 1× |
| C | WX_SUM | 1× |
| D | LHR_INDEX | 1× |
| E | LON_FLY | 1× |

### 5.2 Normalisation

Within each group:
- If your PnL > 0: `score = your_PnL / sum(all_positive_PnLs_in_group)`
- If your PnL < 0: `score = your_PnL / sum(all_negative_PnLs_in_group)` (this is negative)
- Group A scores are multiplied by **4**.
- Final score = sum across all groups.

### 5.3 Strategic Implications

1. **Group A is everything.** 4× multiplier means Markets 1, 3, 5, 7 are highest priority. A loss here is 4× as painful.
2. **Low-competition groups (B, C, D) reward any positive PnL.** If only 3 teams profit in Group B, even £50 captures a large normalised share.
3. **Group E (LON_FLY) is high-value if you have correct pricing.** Most teams will misprice the options package.
4. **Never go negative in Group A.** Reduce position sizes when uncertain rather than risking a loss.

---

## 6. Project Architecture

### 6.1 File Structure

```
imcity-bot/
├── bot_template.py          # PROVIDED — do not modify
├── config.py                # Constants, credentials, URLs
├── main.py                  # Entry point — instantiates and runs the bot
├── bot.py                   # TradingBot(BaseBot) — the main bot class
├── data/
│   ├── thames.py            # Thames tidal data fetcher + harmonic model
│   ├── weather.py           # Weather data fetcher + forecast model
│   └── flights.py           # Heathrow flight data fetcher + schedule model
├── theo/
│   ├── engine.py            # TheoEngine — computes all theos centrally
│   ├── tide_spot.py         # TIDE_SPOT theo: harmonic fit → abs(level) × 1000
│   ├── tide_swing.py        # TIDE_SWING theo: realized + expected strangle sum
│   ├── wx_spot.py           # WX_SPOT theo: forecast T×H at noon
│   ├── wx_sum.py            # WX_SUM theo: realized + expected cumulative sum
│   ├── lhr_count.py         # LHR_COUNT theo: observed + expected remaining
│   ├── lhr_index.py         # LHR_INDEX theo: running flow sum + expected
│   ├── lon_etf.py           # LON_ETF theo: sum of TIDE_SPOT + WX_SPOT + LHR_COUNT
│   └── lon_fly.py           # LON_FLY theo: piecewise options on LON_ETF
├── execution/
│   ├── quoter.py            # Market-making logic: spread management, inventory skew
│   ├── arbitrage.py         # ETF arb engine: LON_ETF vs components
│   └── sniper.py            # Opportunistic mispricing sniper
├── risk/
│   └── manager.py           # Position tracking, limit enforcement, PnL monitoring
└── utils/
    ├── rate_limiter.py       # Token-bucket rate limiter (1 req/sec)
    └── helpers.py            # IOC pattern, order book helpers, logging
```

### 6.2 Data Flow

```
External APIs ──→ data/*.py ──→ TheoEngine ──→ theo per product
                                    │
SSE Stream ──→ on_orderbook() ──→ Book State
                                    │
                              ┌─────┴──────┐
                              │  Execution  │
                              │   Layer     │
                              ├─────────────┤
                              │ • Quoter    │ ← quotes around theo with spread
                              │ • Arb       │ ← LON_ETF vs components
                              │ • Sniper    │ ← lifts mispriced orders
                              └──────┬──────┘
                                     │
                              RiskManager ──→ position limits, PnL tracking
                                     │
                              Exchange (REST)
```

### 6.3 Threading Model

```
Thread 1 (Main):     Main loop — cycles through products, calls quoter/arb/sniper
Thread 2 (SSE):      Background — managed by BaseBot, fires callbacks
Thread 3 (Data):     Background — polls external APIs every N minutes, updates TheoEngine
```

The SSE thread is managed by `BaseBot.start()`. The data polling thread should be a daemon thread started in `TradingBot.__init__` or `TradingBot.start()`.

### 6.4 Key Design Constraints

1. **Rate limit awareness.** Every REST call (send_order, cancel_order, get_orderbook, etc.) counts toward 1 req/sec. The `RateLimiter` must gate ALL outgoing REST calls.
2. **No infinite loops from SSE.** The `on_orderbook` callback fires when YOUR orders change the book. Never unconditionally place orders inside `on_orderbook`.
3. **Thread safety.** The SSE callback runs on Thread 2. The main loop runs on Thread 1. Shared state (positions, theos, book snapshots) must be protected with locks or use atomic updates.
4. **Graceful degradation.** If an external data source fails, widen spreads on affected products rather than crashing. Never trade with stale theos.

---

## 7. Phase 0 — Boilerplate & Infrastructure

> **Goal:** A running bot that connects to the exchange, manages rate limits, tracks positions and order book state, and can place/cancel orders safely. No trading logic yet — just the skeleton.

### 7.1 `config.py`

```python
"""Configuration constants."""
import os

# Exchange
EXCHANGE_URL = os.getenv("CMI_URL", "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/")
USERNAME = os.getenv("CMI_USER", "your_username")
PASSWORD = os.getenv("CMI_PASS", "your_password")

# External data
AERODATABOX_KEY = os.getenv("AERODATABOX_KEY", "")

# Products
SYMBOLS = [
    "TIDE_SPOT", "TIDE_SWING",
    "WX_SPOT", "WX_SUM",
    "LHR_COUNT", "LHR_INDEX",
    "LON_ETF", "LON_FLY",
]

# Group A products (4× weight — highest priority)
GROUP_A = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT", "LON_ETF"]

# Position limits
MAX_POSITION = 100

# Competition timing
COMPETITION_DURATION_HOURS = 24
SETTLEMENT_HOUR = 12  # noon London time

# Rate limit
MAX_REQUESTS_PER_SECOND = 1
```

### 7.2 `utils/rate_limiter.py`

Implement a token-bucket rate limiter. **Every** REST call in the bot must pass through this.

```python
"""Token-bucket rate limiter for exchange REST API."""
import threading
import time


class RateLimiter:
    """Enforces max N requests per second using a token bucket.

    Usage:
        limiter = RateLimiter(max_rps=1)
        limiter.acquire()  # blocks until a token is available
        # ... make request ...
    """

    def __init__(self, max_rps: float = 1.0):
        self._interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last_request = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last_request = time.monotonic()
```

### 7.3 `utils/helpers.py`

```python
"""Shared helper functions."""
from bot_template import BaseBot, OrderBook, Order, OrderRequest, OrderResponse, Side


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
    import math
    if direction == "down":
        return math.floor(price / tick) * tick
    elif direction == "up":
        return math.ceil(price / tick) * tick
    else:
        return round(price / tick) * tick
```

### 7.4 `risk/manager.py`

```python
"""Position and risk management."""
import threading
from config import MAX_POSITION, GROUP_A


class RiskManager:
    """Tracks positions and enforces limits.

    Thread-safe. Updated by the main loop after fills and periodically
    reconciled against the exchange via get_positions().
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._positions: dict[str, int] = {}  # symbol → net position
        self._pnl: dict = {}

    def update_positions(self, positions: dict[str, int]) -> None:
        with self._lock:
            self._positions = dict(positions)

    def get_position(self, symbol: str) -> int:
        with self._lock:
            return self._positions.get(symbol, 0)

    def can_buy(self, symbol: str, volume: int) -> bool:
        return self.get_position(symbol) + volume <= MAX_POSITION

    def can_sell(self, symbol: str, volume: int) -> bool:
        return self.get_position(symbol) - volume >= -MAX_POSITION

    def max_buy_volume(self, symbol: str) -> int:
        return max(0, MAX_POSITION - self.get_position(symbol))

    def max_sell_volume(self, symbol: str) -> int:
        return max(0, MAX_POSITION + self.get_position(symbol))

    def update_pnl(self, pnl: dict) -> None:
        with self._lock:
            self._pnl = pnl

    @property
    def positions(self) -> dict[str, int]:
        with self._lock:
            return dict(self._positions)
```

### 7.5 `bot.py` — Main Bot Shell

```python
"""Main trading bot."""
import math
import threading
import time
import logging
from datetime import datetime, timezone

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Trade, Side
from config import EXCHANGE_URL, USERNAME, PASSWORD, SYMBOLS, MAX_POSITION
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
    # Wrap every inherited REST method that hits the exchange.
    # The bot must call these wrappers instead of the raw inherited methods.

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

    # --- Main Loop (to be filled in later phases) ---

    def run(self) -> None:
        log.info("Starting bot...")
        self.start()  # opens SSE stream
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
        """One iteration of the main trading loop. Filled in by later phases."""
        # Phase 0: just sync positions and sleep
        self.risk.update_positions(self.safe_get_positions())
        log.info(f"Positions: {self.risk.positions}")
        time.sleep(10)
```

### 7.6 `main.py`

```python
"""Entry point."""
from bot import TradingBot

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
```

### 7.7 Phase 0 Acceptance Criteria

- [ ] Bot connects to the test exchange and authenticates
- [ ] SSE stream connects and `on_orderbook` populates `self._books`
- [ ] `on_trades` logs fills
- [ ] `safe_*` wrappers enforce 1 req/sec
- [ ] `RiskManager` tracks positions correctly
- [ ] Main loop runs without crashing
- [ ] No orders are placed (Phase 0 is observation-only)

---

## 8. Phase 1 — ETF Arbitrage Engine

> **Goal:** Exploit the deterministic relationship `LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT`. This is the single most reliable PnL source. Deploy first.

### 8.1 Theory

LON_ETF settles to the sum of its three components. If the market price of LON_ETF diverges from the sum of its components' market prices, you can lock in risk-free profit:

- If `LON_ETF price > TIDE_SPOT + WX_SPOT + LHR_COUNT`: **sell LON_ETF, buy the three components**
- If `LON_ETF price < TIDE_SPOT + WX_SPOT + LHR_COUNT`: **buy LON_ETF, sell the three components**

The profit equals the price difference, locked in at execution, and realised at settlement.

### 8.2 `execution/arbitrage.py`

```python
"""ETF Arbitrage Engine.

Monitors LON_ETF vs (TIDE_SPOT + WX_SPOT + LHR_COUNT).
When a mispricing exceeds the threshold, executes the 4-leg arb.
"""
import logging
from bot_template import OrderBook, OrderRequest, Side
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

        # Case 1: ETF overpriced → sell ETF, buy components
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

        # Case 2: ETF underpriced → buy ETF, sell components
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
        """Compute max volume respecting ±100 position limits on all legs."""
        from config import MAX_POSITION
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
```

### 8.3 Integration into `_main_tick`

Add arb checking to the main loop. The arb engine uses **cached SSE book snapshots** (no REST call needed to read the book) and only makes REST calls to send orders when an opportunity is found.

```python
# In TradingBot._main_tick():
# 1. Read cached books
books = {}
with self._books_lock:
    books = dict(self._books)

# 2. Check arb
arb_orders = self.arb_engine.check_and_generate_orders(
    books, self.risk.positions, max_volume=5
)
if arb_orders:
    for order in arb_orders:
        resp = self.safe_send_ioc(order)  # IOC to avoid resting
        if resp:
            log.info(f"ARB executed: {resp.filled} filled")

# 3. Sync positions after arb
self.risk.update_positions(self.safe_get_positions())
```

### 8.4 Phase 1 Acceptance Criteria

- [ ] `ArbitrageEngine` correctly identifies both over/under-priced ETF scenarios
- [ ] Position limits are respected on all 4 legs
- [ ] Orders are sent as IOC (place + cancel) to avoid resting risk
- [ ] Arb checks run every main loop tick using SSE-cached books (not REST-polled)
- [ ] Fills are logged and positions updated
- [ ] Edge threshold is configurable (start at 2-3, can tighten later)

---

## 9. Phase 2 — Tidal, Weather & Airport Theo Models

> **Goal:** Build accurate theoretical settlement prices for the 5 underlying markets: TIDE_SPOT, WX_SPOT, LHR_COUNT, WX_SUM, and a basic LHR_INDEX. These feed the market-making engine in Phase 3.

### 9.1 `data/thames.py` — Tidal Data + Harmonic Model

```python
"""Thames tidal data fetching and harmonic prediction model."""
import numpy as np
import pandas as pd
import requests
import logging
from datetime import datetime

log = logging.getLogger("data.thames")

THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"


def fetch_thames_readings(limit: int = 200) -> pd.DataFrame:
    """Fetch recent Thames tidal readings from EA API.
    Returns DataFrame: time (London tz), level (mAOD float).
    """
    resp = requests.get(
        f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
        params={"_sorted": "", "_limit": limit},
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    df = pd.DataFrame(items)[["dateTime", "value"]].rename(
        columns={"dateTime": "time", "value": "level"}
    )
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
    df["level"] = df["level"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


class TidalModel:
    """Fits a harmonic model to observed tidal data and predicts future levels.

    Model: h(t) = H0 + Σ Ai × cos(ωi × t + φi)

    Primary constituents:
    - M2 (principal lunar semidiurnal): period ≈ 12.4206 hours
    - S2 (principal solar semidiurnal): period ≈ 12.0000 hours
    - K1 (luni-solar diurnal): period ≈ 23.9345 hours

    The model is fit via least-squares regression on observed data.
    """

    # Known tidal constituent periods in hours
    CONSTITUENTS = {
        "M2": 12.4206,
        "S2": 12.0000,
        "K1": 23.9345,
        "O1": 25.8193,
    }

    def __init__(self):
        self.coeffs: np.ndarray | None = None
        self._t0: datetime | None = None  # reference time for model
        self.observations: pd.DataFrame = pd.DataFrame()

    def update(self, readings: pd.DataFrame) -> None:
        """Fit model to observations. readings must have 'time' and 'level' columns."""
        if len(readings) < 10:
            log.warning("Not enough readings to fit tidal model")
            return

        self.observations = readings.copy()
        self._t0 = readings["time"].iloc[0]

        # Convert times to hours since t0
        t_hours = (readings["time"] - self._t0).dt.total_seconds().values / 3600.0
        y = readings["level"].values

        # Build design matrix: [1, cos(ω1*t), sin(ω1*t), cos(ω2*t), sin(ω2*t), ...]
        X = np.ones((len(t_hours), 1))
        for period in self.CONSTITUENTS.values():
            omega = 2 * np.pi / period
            X = np.column_stack([X, np.cos(omega * t_hours), np.sin(omega * t_hours)])

        # Least-squares fit
        self.coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ self.coeffs
        self.rmse = float(np.sqrt(np.mean(residuals ** 2)))
        log.info(f"Tidal model fit: RMSE = {self.rmse:.4f} mAOD, {len(readings)} obs")

    def predict(self, target_time: datetime) -> float | None:
        """Predict water level (mAOD) at a specific time."""
        if self.coeffs is None or self._t0 is None:
            return None
        t_hours = (target_time - self._t0).total_seconds() / 3600.0
        X = np.array([1.0])
        for period in self.CONSTITUENTS.values():
            omega = 2 * np.pi / period
            X = np.append(X, [np.cos(omega * t_hours), np.sin(omega * t_hours)])
        return float(X @ self.coeffs)

    def predict_series(self, times: list[datetime]) -> list[float]:
        """Predict water levels at multiple times."""
        return [self.predict(t) for t in times]

    @property
    def confidence_m(self) -> float:
        """2-sigma confidence interval in metres."""
        return self.rmse * 2 if self.coeffs is not None else 0.5
```

### 9.2 `theo/tide_spot.py`

```python
"""TIDE_SPOT theo computation."""
from datetime import datetime


def compute_tide_spot_theo(
    tidal_model,  # TidalModel instance
    settlement_time: datetime,
) -> tuple[float, float]:
    """Returns (theo, confidence_half_width) for TIDE_SPOT.

    Settlement = abs(level_mAOD) × 1000
    """
    predicted_level = tidal_model.predict(settlement_time)
    if predicted_level is None:
        return (None, 999.0)

    theo = abs(predicted_level) * 1000.0
    confidence = tidal_model.confidence_m * 1000.0  # convert m → settlement units
    return (round(theo), confidence)
```

### 9.3 `data/weather.py`

```python
"""Weather data fetching."""
import requests
import pandas as pd
import logging

log = logging.getLogger("data.weather")

LONDON_LAT, LONDON_LON = 51.5074, -0.1278


def c_to_f(temp_c: float) -> float:
    return temp_c * 9.0 / 5.0 + 32.0


def fetch_weather(past_steps: int = 96, forecast_steps: int = 96) -> pd.DataFrame:
    """Fetch 15-min weather: past + forecast. Adds temp_f column."""
    variables = (
        "temperature_2m,apparent_temperature,relative_humidity_2m,"
        "precipitation,wind_speed_10m,cloud_cover,visibility"
    )
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LONDON_LAT,
            "longitude": LONDON_LON,
            "minutely_15": variables,
            "past_minutely_15": past_steps,
            "forecast_minutely_15": forecast_steps,
            "timezone": "Europe/London",
        },
    )
    resp.raise_for_status()
    m = resp.json()["minutely_15"]
    df = pd.DataFrame({
        "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
        "temperature_c": m["temperature_2m"],
        "humidity": m["relative_humidity_2m"],
    })
    df["temperature_f"] = df["temperature_c"].apply(c_to_f)
    return df
```

### 9.4 `theo/wx_spot.py`

```python
"""WX_SPOT theo computation."""
import pandas as pd
from datetime import datetime


def compute_wx_spot_theo(
    weather_df: pd.DataFrame,
    settlement_time: datetime,
) -> tuple[float, float]:
    """WX_SPOT = round(temp_F × humidity%) at settlement.

    Uses the forecast row closest to settlement time.
    Returns (theo, confidence).
    """
    if weather_df.empty:
        return (None, 999.0)

    # Find row closest to settlement time
    target = pd.Timestamp(settlement_time).tz_localize("Europe/London") if settlement_time.tzinfo is None else pd.Timestamp(settlement_time)
    idx = (weather_df["time"] - target).abs().idxmin()
    row = weather_df.loc[idx]

    temp_f = row["temperature_f"]
    humidity = row["humidity"]
    theo = round(temp_f * humidity)

    # Confidence: ±2°F temp, ±5% humidity → propagate
    # d(T*H) ≈ H*dT + T*dH
    conf = abs(humidity * 2.0) + abs(temp_f * 5.0)
    return (theo, conf)
```

### 9.5 `theo/wx_sum.py`

```python
"""WX_SUM theo computation."""
import pandas as pd
from datetime import datetime


def compute_wx_sum_theo(
    weather_df: pd.DataFrame,
    competition_start: datetime,
    settlement_time: datetime,
    now: datetime,
) -> tuple[float, float]:
    """WX_SUM = Σ(temp_F × humidity / 100) over all 15-min intervals.

    Split into:
    - realized_sum: intervals that have already passed (observed data)
    - expected_remaining: future intervals (forecast data)
    """
    if weather_df.empty:
        return (None, 999.0)

    start = pd.Timestamp(competition_start)
    end = pd.Timestamp(settlement_time)
    current = pd.Timestamp(now)

    # Filter to competition window
    mask = (weather_df["time"] >= start) & (weather_df["time"] <= end)
    window = weather_df[mask].copy()

    if window.empty:
        return (None, 999.0)

    window["wx_product"] = window["temperature_f"] * window["humidity"] / 100.0

    # Split realized vs forecast
    realized = window[window["time"] <= current]
    forecast = window[window["time"] > current]

    realized_sum = realized["wx_product"].sum() if not realized.empty else 0.0
    forecast_sum = forecast["wx_product"].sum() if not forecast.empty else 0.0

    theo = realized_sum + forecast_sum
    # Confidence shrinks as more is realized
    n_remaining = len(forecast)
    confidence = n_remaining * 3.0  # rough: ±3 per interval uncertainty
    return (round(theo), confidence)
```

### 9.6 `data/flights.py`

```python
"""Heathrow flight data fetching."""
import json
import requests
import pandas as pd
import logging
from datetime import datetime

log = logging.getLogger("data.flights")

AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"


def fetch_flights_range(
    api_key: str,
    from_local: str,
    to_local: str,
    airport: str = "LHR",
) -> dict:
    """Fetch flights in a time range (max 12h span).
    Returns {"arrivals": [...], "departures": [...]}.
    """
    url = (
        f"https://{AERODATABOX_HOST}/flights/airports/iata/"
        f"{airport}/{from_local}/{to_local}?direction=Both"
    )
    resp = requests.get(
        url,
        headers={
            "x-rapidapi-host": AERODATABOX_HOST,
            "x-rapidapi-key": api_key,
        },
    )
    resp.raise_for_status()
    return json.loads(resp.text)


def fetch_full_day(api_key: str, date_str: str = "2026-02-28") -> dict:
    """Fetch full 24h of flights in 2 calls (12h each).
    date_str format: YYYY-MM-DD. Assumes competition starts at noon.
    """
    data1 = fetch_flights_range(
        api_key,
        f"{date_str}T12:00",
        f"{date_str}T23:59",
    )
    # Next day
    parts = date_str.split("-")
    next_day = f"{parts[0]}-{parts[1]}-{int(parts[2])+1:02d}"
    data2 = fetch_flights_range(
        api_key,
        f"{next_day}T00:00",
        f"{next_day}T12:00",
    )
    return {
        "arrivals": data1.get("arrivals", []) + data2.get("arrivals", []),
        "departures": data1.get("departures", []) + data2.get("departures", []),
    }


def count_flights(flight_data: dict) -> int:
    """Total arrivals + departures."""
    return len(flight_data.get("arrivals", [])) + len(flight_data.get("departures", []))
```

### 9.7 `theo/lhr_count.py`

```python
"""LHR_COUNT theo computation."""
from datetime import datetime


def compute_lhr_count_theo(
    observed_flights: int,
    total_scheduled: int,
    hours_elapsed: float,
    total_hours: float = 24.0,
) -> tuple[float, float]:
    """LHR_COUNT = total arrivals + departures over 24h.

    Early: theo ≈ total_scheduled (from AeroDataBox).
    As time passes: theo = observed + (scheduled_remaining × completion_rate).

    The completion_rate adjusts for cancellations observed so far.
    """
    if total_hours <= 0:
        return (total_scheduled, 50.0)

    frac_elapsed = min(hours_elapsed / total_hours, 1.0)

    if frac_elapsed < 0.1:
        # Too early, trust schedule
        return (total_scheduled, 50.0)

    # Estimate completion rate from observed vs expected-by-now
    expected_by_now = total_scheduled * frac_elapsed
    if expected_by_now > 0:
        completion_rate = observed_flights / expected_by_now
    else:
        completion_rate = 1.0

    remaining_scheduled = total_scheduled * (1.0 - frac_elapsed)
    theo = observed_flights + remaining_scheduled * completion_rate

    # Confidence shrinks with time
    confidence = (1.0 - frac_elapsed) * 50.0
    return (round(theo), confidence)
```

### 9.8 `theo/engine.py` — Central Theo Engine

```python
"""Central theo engine. Orchestrates data fetching and theo computation."""
import threading
import time
import logging
from datetime import datetime, timezone, timedelta

from data.thames import fetch_thames_readings, TidalModel
from data.weather import fetch_weather
from data.flights import fetch_full_day, count_flights
from theo.tide_spot import compute_tide_spot_theo
from theo.wx_spot import compute_wx_spot_theo
from theo.wx_sum import compute_wx_sum_theo
from theo.lhr_count import compute_lhr_count_theo
from config import AERODATABOX_KEY

log = logging.getLogger("theo.engine")

# Competition window (update for actual challenge)
COMP_START = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)
SETTLEMENT = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


class TheoEngine:
    """Manages all data sources and computes theos for all products.

    Runs a background thread that periodically refreshes data and
    recomputes theos. The main bot reads theos via get_theo().
    """

    def __init__(self, bot_set_theo_fn):
        """
        Args:
            bot_set_theo_fn: callable(symbol, value, confidence) to push theos to bot
        """
        self._set_theo = bot_set_theo_fn
        self._tidal_model = TidalModel()
        self._weather_df = None
        self._flight_data = None
        self._total_scheduled_flights = 700  # default estimate

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        """Background loop: fetch data, compute theos, sleep, repeat."""
        # Initial fetch
        self._fetch_all_data()
        self._compute_all_theos()

        while not self._stop.is_set():
            self._stop.wait(timeout=120)  # re-fetch every 2 minutes
            if not self._stop.is_set():
                self._fetch_all_data()
                self._compute_all_theos()

    def _fetch_all_data(self) -> None:
        # Thames
        try:
            readings = fetch_thames_readings(limit=200)
            self._tidal_model.update(readings)
            log.info(f"Thames: {len(readings)} readings, RMSE={self._tidal_model.rmse:.4f}")
        except Exception as e:
            log.error(f"Thames fetch failed: {e}")

        # Weather
        try:
            self._weather_df = fetch_weather(past_steps=96, forecast_steps=96)
            log.info(f"Weather: {len(self._weather_df)} rows")
        except Exception as e:
            log.error(f"Weather fetch failed: {e}")

        # Flights (less frequent — API budget is limited)
        if self._flight_data is None:
            try:
                if AERODATABOX_KEY:
                    self._flight_data = fetch_full_day(AERODATABOX_KEY)
                    self._total_scheduled_flights = count_flights(self._flight_data)
                    log.info(f"Flights: {self._total_scheduled_flights} total scheduled")
                else:
                    log.warning("No AeroDataBox API key, using default flight estimate")
            except Exception as e:
                log.error(f"Flight fetch failed: {e}")

    def _compute_all_theos(self) -> None:
        now = datetime.now(timezone.utc)
        hours_elapsed = (now - COMP_START).total_seconds() / 3600.0

        # TIDE_SPOT
        theo, conf = compute_tide_spot_theo(self._tidal_model, SETTLEMENT)
        if theo is not None:
            self._set_theo("TIDE_SPOT", theo, conf)

        # WX_SPOT
        if self._weather_df is not None:
            theo, conf = compute_wx_spot_theo(self._weather_df, SETTLEMENT)
            if theo is not None:
                self._set_theo("WX_SPOT", theo, conf)

        # WX_SUM
        if self._weather_df is not None:
            theo, conf = compute_wx_sum_theo(
                self._weather_df, COMP_START, SETTLEMENT, now
            )
            if theo is not None:
                self._set_theo("WX_SUM", theo, conf)

        # LHR_COUNT
        observed = 0  # TODO: track actual observed flights over time
        theo, conf = compute_lhr_count_theo(
            observed, self._total_scheduled_flights, hours_elapsed
        )
        self._set_theo("LHR_COUNT", theo, conf)

        # LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
        tide = self.get_current_theo("TIDE_SPOT")
        wx = self.get_current_theo("WX_SPOT")
        lhr = self.get_current_theo("LHR_COUNT")
        if all(v is not None for v in [tide, wx, lhr]):
            etf_theo = tide + wx + lhr
            # Confidence is sum of component confidences
            self._set_theo("LON_ETF", round(etf_theo), 999.0)  # refined in Phase 3

        log.info(f"Theos updated. Hours elapsed: {hours_elapsed:.1f}")

    def get_current_theo(self, symbol: str) -> float | None:
        """Read back a theo we've already set. Utility for cross-references."""
        # This is a bit of a hack — in practice the bot stores theos
        # and we'd read from there. For now, store locally too.
        return getattr(self, f"_last_{symbol}", None)
```

### 9.9 Phase 2 Acceptance Criteria

- [ ] Thames data fetches successfully and `TidalModel` produces predictions with RMSE < 0.1 mAOD
- [ ] Weather data fetches, temperature is converted to °F, and WX_SPOT theo is reasonable (e.g. 2000–5000 range)
- [ ] WX_SUM theo accumulates correctly: realized portion + forecast portion
- [ ] LHR_COUNT theo starts at schedule estimate and converges as observations accumulate
- [ ] LON_ETF theo = sum of components
- [ ] TheoEngine runs in background thread, updates every 2 minutes
- [ ] All theos are pushed to the bot via `set_theo()` and are accessible in the main loop

---

## 10. Phase 3 — Market Making on All Products

> **Goal:** Quote two-sided markets (bid + ask) around the theo for every product, with adaptive spreads and inventory management. This is the core money-making loop for Group A (4× weight).

### 10.1 `execution/quoter.py`

```python
"""Market-making quoter with adaptive spreads and inventory skew."""
import math
import logging
from bot_template import OrderRequest, Side
from config import MAX_POSITION

log = logging.getLogger("quoter")


class Quoter:
    """Generates bid/ask quotes around a theoretical price.

    Spread adapts to:
    - Time remaining (tighter as settlement approaches)
    - Confidence interval (tighter when theo is more certain)
    - Inventory (skews quotes to reduce position)
    """

    def __init__(
        self,
        base_spread: float = 10.0,
        min_spread: float = 2.0,
        skew_factor: float = 0.05,
        alpha: float = 0.5,        # spread decay exponent
        default_volume: int = 5,
    ):
        self.base_spread = base_spread
        self.min_spread = min_spread
        self.skew_factor = skew_factor
        self.alpha = alpha
        self.default_volume = default_volume

    def generate_quotes(
        self,
        symbol: str,
        theo: float,
        confidence: float,
        position: int,
        time_remaining_frac: float,   # 1.0 at start, 0.0 at settlement
        tick_size: float = 1.0,
    ) -> list[OrderRequest]:
        """Generate bid and ask OrderRequests for a product.

        Returns 0-2 orders. May skip a side if position limit is hit.
        """
        # Adaptive half-spread
        time_factor = max(time_remaining_frac, 0.01) ** self.alpha
        confidence_factor = max(1.0, confidence / 100.0)
        half_spread = max(
            self.min_spread,
            self.base_spread * time_factor * confidence_factor / 2.0
        )

        # Inventory skew: shift mid towards reducing position
        skew = position * self.skew_factor
        adjusted_mid = theo - skew

        # Compute prices (snap to tick)
        bid_price = math.floor((adjusted_mid - half_spread) / tick_size) * tick_size
        ask_price = math.ceil((adjusted_mid + half_spread) / tick_size) * tick_size

        # Ensure valid spread
        if bid_price >= ask_price:
            ask_price = bid_price + tick_size
        if bid_price <= 0:
            bid_price = tick_size

        orders = []

        # Bid (buy) — only if we can still go longer
        buy_vol = min(self.default_volume, MAX_POSITION - position)
        if buy_vol > 0:
            orders.append(OrderRequest(symbol, bid_price, Side.BUY, buy_vol))

        # Ask (sell) — only if we can still go shorter
        sell_vol = min(self.default_volume, MAX_POSITION + position)
        if sell_vol > 0:
            orders.append(OrderRequest(symbol, ask_price, Side.SELL, sell_vol))

        return orders
```

### 10.2 `execution/sniper.py`

```python
"""Opportunistic sniper: lifts/hits orders that are mispriced vs theo."""
import logging
from bot_template import OrderBook, OrderRequest, Side
from utils.helpers import best_bid, best_ask
from config import MAX_POSITION

log = logging.getLogger("sniper")


class Sniper:
    """Monitors the book for orders far from theo and aggresses them."""

    def __init__(self, min_edge_sigma: float = 2.0):
        """
        Args:
            min_edge_sigma: minimum number of confidence widths away
                            from theo to trigger a snipe.
        """
        self.min_edge_sigma = min_edge_sigma

    def check(
        self,
        symbol: str,
        book: OrderBook,
        theo: float,
        confidence: float,
        position: int,
    ) -> OrderRequest | None:
        """Check for sniping opportunity. Returns an aggressive order or None."""
        if confidence <= 0:
            return None

        threshold = self.min_edge_sigma * confidence

        # Check if best ask is significantly below theo (cheap to buy)
        ask = best_ask(book)
        if ask is not None and theo - ask > threshold:
            vol = min(5, MAX_POSITION - position)
            if vol > 0:
                log.info(f"SNIPE BUY {symbol}: ask={ask}, theo={theo}, edge={theo-ask:.1f}")
                return OrderRequest(symbol, ask, Side.BUY, vol)

        # Check if best bid is significantly above theo (expensive to sell into)
        bid = best_bid(book)
        if bid is not None and bid - theo > threshold:
            vol = min(5, MAX_POSITION + position)
            if vol > 0:
                log.info(f"SNIPE SELL {symbol}: bid={bid}, theo={theo}, edge={bid-theo:.1f}")
                return OrderRequest(symbol, bid, Side.SELL, vol)

        return None
```

### 10.3 Updated Main Loop

```python
def _main_tick(self, products: dict) -> None:
    """Full main loop iteration with arb, quoting, and sniping."""
    now = datetime.now(timezone.utc)
    hours_elapsed = (now - COMP_START).total_seconds() / 3600.0
    time_remaining_frac = max(0.0, 1.0 - hours_elapsed / 24.0)

    # 1. Sync positions
    self.risk.update_positions(self.safe_get_positions())
    positions = self.risk.positions

    # 2. Get cached books
    with self._books_lock:
        books = dict(self._books)

    # 3. Arbitrage check (uses SSE books, no REST needed to read)
    arb_orders = self.arb_engine.check_and_generate_orders(books, positions)
    if arb_orders:
        for order in arb_orders:
            self.safe_send_ioc(order)
        # Re-sync after arb
        self.risk.update_positions(self.safe_get_positions())
        positions = self.risk.positions

    # 4. Cancel all existing quotes
    self.safe_cancel_all_orders()

    # 5. For each product: snipe opportunities + place new quotes
    for symbol in SYMBOLS:
        theo = self.get_theo(symbol)
        if theo is None:
            continue

        book = books.get(symbol)
        pos = positions.get(symbol, 0)
        confidence = self._confidence.get(symbol, 999.0)

        # 5a. Snipe check
        if book is not None:
            snipe_order = self.sniper.check(symbol, book, theo, confidence, pos)
            if snipe_order:
                self.safe_send_ioc(snipe_order)
                pos = self.risk.get_position(symbol)  # may have changed

        # 5b. Place new quotes
        quotes = self.quoter.generate_quotes(
            symbol, theo, confidence, pos, time_remaining_frac,
            tick_size=products[symbol].tickSize,
        )
        for q in quotes:
            self.safe_send_order(q)

    # 6. Log status
    pnl = self.safe_get_pnl()
    log.info(f"Tick complete. Positions: {positions}. PnL: {pnl}")

    # 7. Sleep to pace the loop (adjusted for rate limit budget)
    time.sleep(2)
```

### 10.4 Phase 3 Acceptance Criteria

- [ ] Quoter generates valid bid/ask around theo for all products
- [ ] Spreads tighten over time (test with simulated `time_remaining_frac` values)
- [ ] Inventory skew shifts quotes to reduce position (long → lower mid, short → higher mid)
- [ ] Position limits are never exceeded
- [ ] Sniper detects and lifts/hits mispriced orders via IOC
- [ ] Main loop completes within rate limit budget
- [ ] Group A products (TIDE_SPOT, WX_SPOT, LHR_COUNT, LON_ETF) are always quoted when theo is available
- [ ] Bot does not place orders when theo is None (stale data guard)

---

## 11. Phase 4 — Derivatives Pricing (TIDE_SWING, LHR_INDEX, LON_FLY)

> **Goal:** Add pricing models for the three complex derivative products. These are lower priority but can capture significant normalised score in their respective groups (B, D, E).

### 11.1 `theo/tide_swing.py` — TIDE_SWING (Group B)

```python
"""TIDE_SWING theo: realized + expected strangle sum on 15-min tidal diffs."""
import numpy as np
from datetime import datetime, timedelta


def strangle_value(diff_cm: float) -> float:
    """Strangle payoff: Put@20cm / Call@25cm.
    diff_cm is the absolute 15-min difference in water level in centimetres.
    """
    return max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)


def compute_tide_swing_theo(
    tidal_model,           # TidalModel instance
    observed_diffs_cm: list[float],   # already-observed 15-min abs diffs in cm
    competition_start: datetime,
    settlement_time: datetime,
    now: datetime,
) -> tuple[float, float]:
    """
    TIDE_SWING = Σ strangle_value(diff_cm_i) for all 96 15-min intervals.

    Split into realized (observed) and forecast (predicted from tidal model).
    """
    # Realized portion
    realized_sum = sum(strangle_value(d) for d in observed_diffs_cm)

    # Forecast remaining intervals
    # Generate predicted levels at 15-min intervals from now to settlement
    forecast_sum = 0.0
    n_remaining = 0

    t = now
    prev_level = tidal_model.predict(t)
    while t < settlement_time:
        t_next = t + timedelta(minutes=15)
        if t_next > settlement_time:
            break
        next_level = tidal_model.predict(t_next)
        if prev_level is not None and next_level is not None:
            diff_cm = abs(next_level - prev_level) * 100.0  # m → cm
            forecast_sum += strangle_value(diff_cm)
            n_remaining += 1
        prev_level = next_level
        t = t_next

    theo = realized_sum + forecast_sum
    confidence = n_remaining * 2.0  # rough ±2 per remaining interval
    return (round(theo), confidence)
```

### 11.2 `theo/lhr_index.py` — LHR_INDEX (Group D)

```python
"""LHR_INDEX theo: abs(Σ flow_metric) over 30-min intervals."""
from datetime import datetime


def flow_metric(arrivals: int, departures: int) -> float:
    """Single 30-min interval contribution: 100 × (arr - dep) / max(arr + dep, 1)"""
    total = arrivals + departures
    if total == 0:
        return 0.0
    return 100.0 * (arrivals - departures) / total


def compute_lhr_index_theo(
    observed_intervals: list[tuple[int, int]],  # list of (arrivals, departures) per 30-min
    expected_remaining: list[tuple[int, int]],   # from schedule
) -> tuple[float, float]:
    """
    LHR_INDEX = abs(Σ flow_metric_i) over all 48 30-min intervals.

    The final absolute value makes this path-dependent.
    """
    running_sum = sum(flow_metric(a, d) for a, d in observed_intervals)
    expected_sum = sum(flow_metric(a, d) for a, d in expected_remaining)

    # Expected settlement
    total_sum = running_sum + expected_sum
    theo = abs(total_sum)

    # Confidence: higher when running sum is far from zero (abs value less sensitive)
    # Lower when running sum is near zero (could flip sign)
    n_remaining = len(expected_remaining)
    base_conf = n_remaining * 5.0
    # If the sum is near zero, uncertainty is higher
    if abs(total_sum) < base_conf:
        confidence = base_conf * 2.0  # extra uncertain
    else:
        confidence = base_conf

    return (round(theo), confidence)
```

### 11.3 `theo/lon_fly.py` — LON_FLY (Group E)

```python
"""LON_FLY theo: options package on LON_ETF."""
import numpy as np


def lon_fly_payoff(s: float) -> float:
    """Piecewise payoff of the LON_FLY options package given ETF settlement s.

    Portfolio: Long 2× Put(6200), Long 1× Call(6200),
               Short 2× Call(6600), Long 3× Call(7000)

    Regimes:
        S < 6200:           12400 - 2S
        6200 ≤ S < 6600:    S - 6200
        6600 ≤ S < 7000:    7000 - S
        S ≥ 7000:           2S - 14000
    """
    return (
        2.0 * max(0.0, 6200.0 - s)
        + 1.0 * max(0.0, s - 6200.0)
        - 2.0 * max(0.0, s - 6600.0)
        + 3.0 * max(0.0, s - 7000.0)
    )


def compute_lon_fly_theo(
    etf_theo: float,
    etf_confidence: float,
) -> tuple[float, float]:
    """Compute LON_FLY theo by integrating payoff over ETF distribution.

    Models ETF settlement as Gaussian(etf_theo, etf_confidence/2).
    When confidence is tight, this collapses to point estimate.
    """
    sigma = max(etf_confidence / 2.0, 1.0)

    # Monte Carlo integration (fast enough, ~10k samples)
    np.random.seed(42)
    samples = np.random.normal(etf_theo, sigma, size=10000)
    payoffs = np.array([lon_fly_payoff(s) for s in samples])

    theo = float(np.mean(payoffs))
    confidence = float(np.std(payoffs))
    return (round(theo), confidence)
```

### 11.4 Wiring Phase 4 into TheoEngine

Add to `TheoEngine._compute_all_theos()`:

```python
# TIDE_SWING
from theo.tide_swing import compute_tide_swing_theo
# (need to track observed diffs — add to TheoEngine state)
theo, conf = compute_tide_swing_theo(
    self._tidal_model, self._observed_tide_diffs_cm,
    COMP_START, SETTLEMENT, now
)
if theo is not None:
    self._set_theo("TIDE_SWING", theo, conf)

# LHR_INDEX
from theo.lhr_index import compute_lhr_index_theo
theo, conf = compute_lhr_index_theo(
    self._observed_flow_intervals, self._expected_flow_intervals
)
self._set_theo("LHR_INDEX", theo, conf)

# LON_FLY
from theo.lon_fly import compute_lon_fly_theo
etf_theo = self.get_current_theo("LON_ETF")
if etf_theo is not None:
    theo, conf = compute_lon_fly_theo(etf_theo, etf_confidence)
    self._set_theo("LON_FLY", theo, conf)
```

### 11.5 Phase 4 Acceptance Criteria

- [ ] `strangle_value(9) == 11` and `strangle_value(33) == 8` and `strangle_value(22) == 0`
- [ ] `lon_fly_payoff(6000) == 400`, `lon_fly_payoff(6400) == 200`, `lon_fly_payoff(7000) == 0`, `lon_fly_payoff(7200) == 400`
- [ ] TIDE_SWING theo combines realized observations + predicted future intervals
- [ ] LHR_INDEX handles the absolute-value-at-end correctly and widens confidence near zero
- [ ] LON_FLY correctly integrates over ETF uncertainty distribution
- [ ] All three products are quoted by the market-making engine

---

## 12. Operational Playbook

### 12.1 Pre-Competition Checklist

- [ ] Test bot against test exchange (TEST_URL) — confirm it connects, quotes, and fills
- [ ] Verify all data sources are returning data
- [ ] Pre-fetch tidal predictions and Heathrow schedule
- [ ] Set `COMP_START` and `SETTLEMENT` datetimes correctly
- [ ] Set challenge exchange URL and credentials
- [ ] Verify unit tests pass on all settlement formulae

### 12.2 Race-Day Phases

| Hours | Phase | Actions |
|-------|-------|---------|
| 0–1 | **Boot & Observe** | Start bot. Let TheoEngine populate theos. Start arb engine. Wide spreads. |
| 1–6 | **Calibrate** | Tidal model gains accuracy. Weather forecast validates. Tighten spreads. |
| 6–18 | **Cruise** | Full market-making on all products. Arb running. Sniper active. |
| 18–22 | **Tighten** | Confidence intervals should be small. Tighten spreads aggressively. |
| 22–24 | **Convergence** | Switch to directional: if any market is mispriced vs your near-certain theo, take max position. Close unintended risk in final 30 min. |

### 12.3 Emergency Procedures

**Data source goes down:**
- Set confidence to 999.0 for affected products → spreads widen automatically
- Log a warning
- Never quote with stale theos

**Bot crash:**
- On restart: cancel all resting orders immediately (`safe_cancel_all_orders`)
- Re-sync positions
- Resume normal operation

**Position limit hit:**
- Quoter automatically stops quoting the limited side
- Inventory skew should be preventing this — if it's happening, increase `skew_factor`

**Exchange rate limit violation:**
- The rate limiter should prevent this
- If it happens anyway, backoff exponentially and log

### 12.4 Parameter Tuning Guide

| Parameter | File | Start Value | When to Adjust |
|-----------|------|-------------|---------------|
| `base_spread` | `quoter.py` | 10 | Decrease if capturing no spread; increase if getting adversely selected |
| `min_spread` | `quoter.py` | 2 | Only decrease in final hours when theos are near-certain |
| `skew_factor` | `quoter.py` | 0.05 | Increase if inventory builds up; decrease if it's too hard to get filled |
| `default_volume` | `quoter.py` | 5 | Increase for liquid products; keep low for illiquid |
| `min_edge` | `arbitrage.py` | 2.0 | Decrease to capture more arb; increase if experiencing leg risk |
| `min_edge_sigma` | `sniper.py` | 2.0 | Decrease for more aggressive sniping; increase to be more selective |
| `alpha` | `quoter.py` | 0.5 | Controls spread decay speed. Higher = slower decay |

### 12.5 Key Formulae Quick Reference

| Product | Settlement | Units |
|---------|-----------|-------|
| TIDE_SPOT | `abs(mAOD) × 1000` | mm |
| TIDE_SWING | `Σ [max(0, 20-d) + max(0, d-25)]` where d = abs diff in cm | cm strangle |
| WX_SPOT | `round(temp_F × humidity%)` | — |
| WX_SUM | `Σ (temp_F × humidity% / 100)` per 15-min | — |
| LHR_COUNT | `arrivals + departures` | flights |
| LHR_INDEX | `abs(Σ [100 × (arr-dep)/max(arr+dep,1)])` per 30-min | — |
| LON_ETF | `TIDE_SPOT + WX_SPOT + LHR_COUNT` | — |
| LON_FLY | `2×P(6200) + C(6200) - 2×C(6600) + 3×C(7000)` on ETF | — |

### 12.6 Scoring Groups (reminder)

| Group | Markets | Weight | Strategy Priority |
|-------|---------|--------|-------------------|
| **A** | TIDE_SPOT, WX_SPOT, LHR_COUNT, LON_ETF | **4×** | HIGHEST — protect at all costs |
| B | TIDE_SWING | 1× | Medium — exploit tidal-cycle mispricing |
| C | WX_SUM | 1× | Medium — straightforward accumulation |
| D | LHR_INDEX | 1× | Cautious — tricky near zero |
| E | LON_FLY | 1× | High-conviction if pricing is correct |
