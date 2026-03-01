"""Central theo engine. Fetches external data and computes fair values for all 8 products.

Runs in a background daemon thread, polling APIs every THEO_POLL_INTERVAL seconds.
Thread-safe reads via get_theo() and get_confidence().
"""
import threading
import time
import logging
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from data.thames import fetch_thames_readings, TidalModel
from data.weather import fetch_weather
from data.flights import (
    fetch_heathrow_flights, estimate_total_lhr_count,
    bin_flights_30min, compute_lhr_index,
)
from config import COMP_START, SETTLEMENT_TIME

log = logging.getLogger("theo.engine")

THEO_POLL_INTERVAL = 120  # seconds between API refreshes


def lon_fly_payoff(s: float) -> float:
    """Compute LON_FLY settlement given LON_ETF settlement value s."""
    return (
        2.0 * max(0.0, 6200.0 - s)
        + 1.0 * max(0.0, s - 6200.0)
        - 2.0 * max(0.0, s - 6600.0)
        + 3.0 * max(0.0, s - 7000.0)
    )


def strangle_value(diff_cm: float) -> float:
    """Compute strangle payoff for a single 15-min tidal diff."""
    return max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)


class TheoEngine:
    """Manages all data sources and computes theos for all products."""

    def __init__(self):
        self._lock = threading.Lock()
        self._theos: dict[str, float | None] = {}
        self._confidence: dict[str, float] = {}

        self._tidal_model = TidalModel()
        self._weather_df: pd.DataFrame | None = None

        # Flight data from OpenSky + FR24
        self._flight_data: dict | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start background polling thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="theo-engine")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def get_theo(self, symbol: str) -> float | None:
        with self._lock:
            return self._theos.get(symbol)

    def get_confidence(self, symbol: str) -> float:
        with self._lock:
            return self._confidence.get(symbol, 999.0)

    def get_all_theos(self) -> dict[str, float | None]:
        with self._lock:
            return dict(self._theos)

    def _set_theo(self, symbol: str, value: float, confidence: float) -> None:
        with self._lock:
            self._theos[symbol] = value
            self._confidence[symbol] = confidence

    # --- Background loop ---

    def _run(self) -> None:
        """Fetch data, compute theos, sleep, repeat."""
        log.info("TheoEngine started")
        self._fetch_and_compute()
        while not self._stop.is_set():
            self._stop.wait(timeout=THEO_POLL_INTERVAL)
            if not self._stop.is_set():
                self._fetch_and_compute()

    def _fetch_and_compute(self) -> None:
        self._fetch_thames()
        self._fetch_weather()
        self._fetch_flights()
        self._compute_all_theos()

    # --- Data fetching ---

    def _fetch_thames(self) -> None:
        try:
            readings = fetch_thames_readings(limit=500)
            if not readings.empty:
                self._tidal_model.update(readings)
        except Exception as e:
            log.error(f"Thames fetch failed: {e}")

    def _fetch_weather(self) -> None:
        try:
            self._weather_df = fetch_weather(past_steps=96, forecast_steps=96)
        except Exception as e:
            log.error(f"Weather fetch failed: {e}")

    def _fetch_flights(self) -> None:
        try:
            now = datetime.now(timezone.utc)
            self._flight_data = fetch_heathrow_flights(COMP_START, now)
            dep_count = self._flight_data.get("dep_count", 0)
            arr_count = self._flight_data.get("arr_count", 0)
            log.info(
                f"Heathrow flights: {dep_count} deps + {arr_count} arrs "
                f"= {dep_count + arr_count} total"
            )
        except Exception as e:
            log.error(f"Flight data fetch failed: {e}")

    # --- Theo computations ---

    def _compute_all_theos(self) -> None:
        now = datetime.now(timezone.utc)
        settlement = SETTLEMENT_TIME
        hours_elapsed = (now - COMP_START).total_seconds() / 3600.0
        total_hours = (SETTLEMENT_TIME - COMP_START).total_seconds() / 3600.0

        # TIDE_SPOT
        self._compute_tide_spot(settlement)

        # WX_SPOT
        self._compute_wx_spot(settlement)

        # LHR_COUNT
        self._compute_lhr_count(hours_elapsed, total_hours)

        # LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT
        self._compute_lon_etf()

        # LON_FLY (from LON_ETF distribution)
        self._compute_lon_fly()

        # WX_SUM
        self._compute_wx_sum(now)

        # TIDE_SWING
        self._compute_tide_swing(now, settlement)

        # LHR_INDEX (low confidence)
        self._compute_lhr_index()

        theos = self.get_all_theos()
        log.info(f"Theos: {theos}")

    def _compute_tide_spot(self, settlement: datetime) -> None:
        predicted = self._tidal_model.predict(settlement)
        if predicted is not None:
            theo = abs(predicted) * 1000.0
            conf = self._tidal_model.confidence_m * 1000.0
            self._set_theo("TIDE_SPOT", round(theo), conf)

    def _compute_wx_spot(self, settlement: datetime) -> None:
        if self._weather_df is None or self._weather_df.empty:
            return
        target = pd.Timestamp(settlement)
        if target.tzinfo is None:
            target = target.tz_localize("UTC").tz_convert("Europe/London")
        else:
            target = target.tz_convert("Europe/London")

        idx = (self._weather_df["time"] - target).abs().idxmin()
        row = self._weather_df.loc[idx]
        temp_f = row["temperature_f"]
        humidity = row["humidity"]
        theo = round(temp_f * humidity)
        # Confidence: ±2°F temp, ±5% humidity
        conf = abs(humidity * 2.0) + abs(temp_f * 5.0)
        self._set_theo("WX_SPOT", theo, conf)

    def _compute_lhr_count(self, hours_elapsed: float, total_hours: float) -> None:
        now = datetime.now(timezone.utc)
        if self._flight_data is not None:
            theo, conf = estimate_total_lhr_count(
                self._flight_data, COMP_START, SETTLEMENT_TIME, now,
            )
            self._set_theo("LHR_COUNT", theo, conf)
        else:
            # No flight data yet — use conservative fallback
            self._set_theo("LHR_COUNT", 1300, 200.0)

    def _compute_lon_etf(self) -> None:
        tide = self.get_theo("TIDE_SPOT")
        wx = self.get_theo("WX_SPOT")
        lhr = self.get_theo("LHR_COUNT")
        if all(v is not None for v in [tide, wx, lhr]):
            etf_theo = tide + wx + lhr
            tide_conf = self.get_confidence("TIDE_SPOT")
            wx_conf = self.get_confidence("WX_SPOT")
            lhr_conf = self.get_confidence("LHR_COUNT")
            conf = math.sqrt(tide_conf**2 + wx_conf**2 + lhr_conf**2)
            self._set_theo("LON_ETF", round(etf_theo), conf)

    def _compute_lon_fly(self) -> None:
        etf_theo = self.get_theo("LON_ETF")
        etf_conf = self.get_confidence("LON_ETF")
        if etf_theo is None:
            return

        # Monte Carlo: sample from N(etf_theo, (conf/2)^2) and average payoff
        sigma = max(etf_conf / 2.0, 50.0)
        samples = np.random.normal(etf_theo, sigma, 5000)
        payoffs = [lon_fly_payoff(s) for s in samples]
        theo = round(float(np.mean(payoffs)))
        conf = float(np.std(payoffs))
        self._set_theo("LON_FLY", theo, conf)

    def _compute_wx_sum(self, now: datetime) -> None:
        if self._weather_df is None or self._weather_df.empty:
            return

        start = pd.Timestamp(COMP_START)
        end = pd.Timestamp(SETTLEMENT_TIME)
        current = pd.Timestamp(now)

        # Ensure timezone consistency
        if start.tzinfo is None:
            start = start.tz_localize("UTC").tz_convert("Europe/London")
        else:
            start = start.tz_convert("Europe/London")
        if end.tzinfo is None:
            end = end.tz_localize("UTC").tz_convert("Europe/London")
        else:
            end = end.tz_convert("Europe/London")
        if current.tzinfo is None:
            current = current.tz_localize("UTC").tz_convert("Europe/London")
        else:
            current = current.tz_convert("Europe/London")

        mask = (self._weather_df["time"] >= start) & (self._weather_df["time"] <= end)
        window = self._weather_df[mask].copy()
        if window.empty:
            return

        window["wx_product"] = window["temperature_f"] * window["humidity"] / 100.0
        realized = window[window["time"] <= current]
        forecast = window[window["time"] > current]

        realized_sum = realized["wx_product"].sum() if not realized.empty else 0.0
        forecast_sum = forecast["wx_product"].sum() if not forecast.empty else 0.0

        theo = round(realized_sum + forecast_sum)
        n_remaining = len(forecast)
        conf = n_remaining * 3.0
        self._set_theo("WX_SUM", theo, conf)

    def _compute_tide_swing(self, now: datetime, settlement: datetime) -> None:
        """TIDE_SWING = sum of strangle payoffs on 96 15-min tidal diffs."""
        if self._tidal_model.coeffs is None:
            return

        start = COMP_START
        # Generate all 96+1 timestamps at 15-min intervals
        times = [start + timedelta(minutes=15 * i) for i in range(97)]

        # Predict levels at all times
        levels = self._tidal_model.predict_series(times)
        if any(l is None for l in levels):
            return

        # Compute 15-min diffs in cm and strangle payoffs
        total_payoff = 0.0
        for i in range(96):
            diff_m = abs(levels[i + 1] - levels[i])
            diff_cm = diff_m * 100.0
            total_payoff += strangle_value(diff_cm)

        conf = 96.0 * self._tidal_model.confidence_m * 100.0 * 0.1  # rough
        self._set_theo("TIDE_SWING", round(total_payoff), max(conf, 50.0))

    def _compute_lhr_index(self) -> None:
        now = datetime.now(timezone.utc)
        if self._flight_data is not None:
            intervals = bin_flights_30min(
                self._flight_data, COMP_START, SETTLEMENT_TIME, now,
            )
            theo, conf = compute_lhr_index(intervals)
            self._set_theo("LHR_INDEX", theo, conf)
        else:
            self._set_theo("LHR_INDEX", 100, 200.0)
