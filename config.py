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

# Arbitrage
MIN_ARB_EDGE = 15.0  # minimum ticks of edge to trigger arb (was 2.0, raised to cover scratch costs)
MIN_BOOK_DEPTH = 3   # minimum contracts available on each leg to attempt arb

# Scratch
MAX_SCRATCH_COST = 50.0  # skip scratch if estimated cost exceeds this

# Rate limit
MAX_REQUESTS_PER_SECOND = 1
