"""Configuration constants."""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
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

# Group A products (4x weight — highest priority)
GROUP_A = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT", "LON_ETF"]

# Position limits
MAX_POSITION = 100

# Competition timing
COMPETITION_DURATION_HOURS = 24
SETTLEMENT_HOUR = 12  # noon London time
COMP_START = datetime(2026, 2, 28, 14, 30, tzinfo=timezone.utc)     # Fri Feb 28 2:30 PM UTC
SETTLEMENT_TIME = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)  # Sat March 1 12:00 PM UTC
# Settlement counts flights over 24h ending at SETTLEMENT_TIME (not from COMP_START)
SETTLEMENT_WINDOW_START = datetime(2026, 2, 28, 12, 0, tzinfo=timezone.utc)  # 24h before settlement

# Arbitrage
MIN_ARB_EDGE = 1.5   # tight to capture more arb opportunities
MIN_BOOK_DEPTH = 2   # reduced to trade in thinner books

# Scratch
MAX_SCRATCH_COST = 500.0  # skip scratch if estimated cost exceeds this (high to avoid unhedged risk)

# Rate limit
MAX_REQUESTS_PER_SECOND = 16

# Strategy engine
ARB_COOLDOWN_SECONDS = 0.5     # brief pause after arb IOC burst (plenty of rate budget)
STRATEGY_WARMUP_TICKS = 3      # min price updates before trading (faster start)
REPRICE_THRESHOLD = 2.5        # ticks of drift before repricing GTC order (wider to reduce cancel churn)
STALE_ORDER_SECONDS = 60.0     # force-reprice after this many seconds

# Theo-based aggressive trading
AGGRESSIVE_THRESHOLD = 0.008   # 0.8% deviation from theo triggers aggressive IOC
AGGRESSIVE_ORDER_SIZE = 5      # contracts per aggressive IOC order

# Stop-loss
STOP_LOSS_PNL = 250_000.0     # halt trading if PnL drops below this
STOP_LOSS_CHECK_INTERVAL = 30  # seconds between PnL checks for stop-loss
