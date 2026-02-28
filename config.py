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

# Group A products (4x weight — highest priority)
GROUP_A = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT", "LON_ETF"]

# Position limits
MAX_POSITION = 100

# Competition timing
COMPETITION_DURATION_HOURS = 24
SETTLEMENT_HOUR = 12  # noon London time

# Rate limit
MAX_REQUESTS_PER_SECOND = 1
