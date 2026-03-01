"""Heathrow flight data from the official Heathrow Airport API.

Fetches real scheduled and actual flight data directly from Heathrow's
public API (api-dp-prod.dp.heathrow.com). This gives 100% accurate data
including flight status (departed, landed, cancelled), exact scheduled times,
and codeshare deduplication.

No API key required — just needs Origin/Referer headers.
"""
import logging
import time
from datetime import datetime, timezone, timedelta

import requests

log = logging.getLogger("data.flights")

HEATHROW_API_BASE = "https://api-dp-prod.dp.heathrow.com/pihub/flights"
HEATHROW_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Origin": "https://www.heathrow.com",
    "Referer": "https://www.heathrow.com/",
}

# Cache to avoid redundant API calls within poll interval
_cache: dict[str, tuple[float, list]] = {}
CACHE_TTL = 120.0  # seconds


# --- Heathrow API ---

def fetch_heathrow_official(date_str: str, direction: str) -> list[dict]:
    """Fetch flights from Heathrow's official API.

    Args:
        date_str: Date in YYYY-MM-DD format.
        direction: 'departures' or 'arrivals'.

    Returns list of raw flight dicts.
    """
    cache_key = f"{date_str}_{direction}"
    now = time.monotonic()
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_data

    url = f"{HEATHROW_API_BASE}/{direction}?date={date_str}"
    try:
        resp = requests.get(url, headers=HEATHROW_HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            _cache[cache_key] = (now, data)
            log.info(f"Heathrow {direction} {date_str}: {len(data)} flights")
            return data
        log.warning(f"Heathrow API {direction} {date_str}: HTTP {resp.status_code}")
    except Exception as e:
        log.error(f"Heathrow API {direction} {date_str} failed: {e}")

    # Return cached data on failure
    if cache_key in _cache:
        return _cache[cache_key][1]
    return []


def parse_operating_flights(
    flights: list[dict],
    direction: str,
) -> list[tuple[str, datetime]]:
    """Parse operating flights from Heathrow API response.

    Filters out codeshare marketing flights and cancelled flights.
    Returns list of (flight_id, scheduled_utc_datetime).
    """
    port_type = "ORIGIN" if direction == "departures" else "DESTINATION"
    result = []

    for flight in flights:
        fs = flight.get("flightService", {})

        # Skip marketing codeshares (duplicates of operating flights)
        if fs.get("codeShareStatus") == "CODESHARE_MARKETING_FLIGHT":
            continue

        # Skip cancelled flights
        movements = fs.get("aircraftMovement", {}).get("aircraftMovementStatus", [])
        if movements and movements[0].get("statusCode") == "CX":
            continue

        # Extract scheduled time
        flight_id = fs.get("iataFlightIdentifier", "")
        ports = fs.get("aircraftMovement", {}).get("route", {}).get("portsOfCall", [])
        sched_utc = None
        for port in ports:
            if port.get("portOfCallType") == port_type:
                utc_str = (
                    port.get("operatingTimes", {})
                    .get("scheduled", {})
                    .get("utc", "")
                )
                if utc_str:
                    sched_utc = datetime.fromisoformat(utc_str).replace(
                        tzinfo=timezone.utc
                    )
                break

        if sched_utc is not None and flight_id:
            result.append((flight_id, sched_utc))

    return result


# --- Main integration functions ---

def fetch_heathrow_flights(
    comp_start: datetime,
    now: datetime,
) -> dict:
    """Fetch all Heathrow flight data covering the competition window.

    Fetches both dates (competition spans midnight) and filters to window.
    Returns dict compatible with theo/engine.py interface.
    """
    from config import SETTLEMENT_TIME

    # Determine which dates to fetch
    dates = set()
    dates.add(comp_start.strftime("%Y-%m-%d"))
    dates.add(SETTLEMENT_TIME.strftime("%Y-%m-%d"))
    dates = sorted(dates)

    # Fetch and parse all flights
    all_departures: dict[str, datetime] = {}
    all_arrivals: dict[str, datetime] = {}

    for date_str in dates:
        raw_deps = fetch_heathrow_official(date_str, "departures")
        parsed_deps = parse_operating_flights(raw_deps, "departures")
        for fid, sched in parsed_deps:
            if comp_start <= sched <= SETTLEMENT_TIME:
                all_departures[fid] = sched  # dedup by flight ID

        raw_arrs = fetch_heathrow_official(date_str, "arrivals")
        parsed_arrs = parse_operating_flights(raw_arrs, "arrivals")
        for fid, sched in parsed_arrs:
            if comp_start <= sched <= SETTLEMENT_TIME:
                all_arrivals[fid] = sched

    dep_count = len(all_departures)
    arr_count = len(all_arrivals)
    log.info(
        f"Heathrow flights in window: {dep_count} deps + {arr_count} arrs "
        f"= {dep_count + arr_count} total"
    )

    return {
        "departures": all_departures,
        "arrivals": all_arrivals,
        "dep_count": dep_count,
        "arr_count": arr_count,
    }


def estimate_total_lhr_count(
    flight_data: dict,
    comp_start: datetime,
    settlement: datetime,
    now: datetime,
) -> tuple[int, float]:
    """Compute LHR_COUNT from Heathrow official data.

    Returns (total_count, confidence).
    """
    dep_count = flight_data.get("dep_count", 0)
    arr_count = flight_data.get("arr_count", 0)
    total = dep_count + arr_count

    if total == 0:
        return (1000, 200.0)  # fallback if API fails

    # Small discount for potential future cancellations
    departures = flight_data.get("departures", {})
    arrivals = flight_data.get("arrivals", {})

    past_deps = sum(1 for t in departures.values() if t <= now)
    past_arrs = sum(1 for t in arrivals.values() if t <= now)
    future_deps = dep_count - past_deps
    future_arrs = arr_count - past_arrs

    realized = past_deps + past_arrs
    scheduled = future_deps + future_arrs
    # ~3% cancellation rate for future flights
    theo = int(realized + scheduled * 0.97)

    # Confidence: very tight since this is official data
    conf = max(5.0, scheduled * 0.3)

    log.info(
        f"LHR_COUNT: {theo} (realized={realized}, scheduled={scheduled}, "
        f"deps={dep_count}, arrs={arr_count})"
    )
    return (theo, conf)


# --- 30-minute binning for LHR_INDEX ---

def bin_flights_30min(
    flight_data: dict,
    comp_start: datetime,
    settlement: datetime,
    now: datetime,
) -> list[tuple[int, int]]:
    """Bin arrivals and departures into 30-minute intervals for LHR_INDEX.

    Returns list of 48 tuples: (arrivals, departures) per interval.
    """
    total_intervals = 48
    interval_seconds = 1800  # 30 minutes

    departures = flight_data.get("departures", {})
    arrivals = flight_data.get("arrivals", {})

    arr_bins = [0] * total_intervals
    dep_bins = [0] * total_intervals

    for fid, t in departures.items():
        idx = int((t - comp_start).total_seconds() / interval_seconds)
        if 0 <= idx < total_intervals:
            dep_bins[idx] += 1

    for fid, t in arrivals.items():
        idx = int((t - comp_start).total_seconds() / interval_seconds)
        if 0 <= idx < total_intervals:
            arr_bins[idx] += 1

    return list(zip(arr_bins, dep_bins))


def compute_lhr_index(intervals: list[tuple[int, int]]) -> tuple[float, float]:
    """Compute LHR_INDEX from 30-min interval data.

    LHR_INDEX = abs(Σ [100 × (arr_i - dep_i) / max(arr_i + dep_i, 1)])

    Returns (index_value, confidence).
    """
    flow_sum = 0.0
    for arr, dep in intervals:
        total = arr + dep
        if total > 0:
            flow = 100.0 * (arr - dep) / max(total, 1.0)
        else:
            flow = 0.0
        flow_sum += flow

    index_val = abs(flow_sum)

    # Confidence: tight for official data, wider near zero (sign-flip risk)
    if abs(flow_sum) < 50:
        conf = 80.0
    elif abs(flow_sum) < 200:
        conf = 40.0
    else:
        conf = 20.0

    log.info(f"LHR_INDEX: {index_val:.0f} (raw_flow={flow_sum:.1f}, conf={conf:.0f})")
    return (round(index_val), conf)
