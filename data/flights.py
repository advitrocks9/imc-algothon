"""Heathrow flight data from AeroDataBox (RapidAPI).

Fetches flight schedules from AeroDataBox and caches to disk since
flight schedules are mostly static during the competition window.
On subsequent polls, reads from the cache file instead of calling the API.

Uses withCodeshared=false to count operating flights only (physical movements).
"""
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta

import requests

log = logging.getLogger("data.flights")

AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"
FLIGHT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "flight_cache.json")


# --- AeroDataBox API ---

def _fetch_flights_range(
    api_key: str,
    from_local: str,
    to_local: str,
    airport: str = "LHR",
    max_retries: int = 3,
) -> dict:
    """Fetch flights in a time range (max 12h span).

    Returns {"arrivals": [...], "departures": [...]}.
    Retries on 429 with exponential backoff.
    """
    url = (
        f"https://{AERODATABOX_HOST}/flights/airports/iata/"
        f"{airport}/{from_local}/{to_local}?direction=Both&withCodeshared=false"
    )
    for attempt in range(max_retries + 1):
        resp = requests.get(
            url,
            headers={
                "x-rapidapi-host": AERODATABOX_HOST,
                "x-rapidapi-key": api_key,
            },
            timeout=30,
        )
        if resp.status_code == 429 and attempt < max_retries:
            backoff = 2.0 * (2 ** attempt)  # 2s, 4s, 8s
            log.warning(f"AeroDataBox 429 (attempt {attempt+1}), retrying in {backoff:.0f}s")
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def _save_cache(data: dict, comp_start: datetime, settlement: datetime) -> None:
    """Save flight data to disk cache with window metadata for validation."""
    try:
        data["_cache_window_start"] = comp_start.isoformat()
        data["_cache_window_end"] = settlement.isoformat()
        with open(FLIGHT_CACHE_FILE, "w") as f:
            json.dump(data, f)
        log.info(f"Flight cache saved to {FLIGHT_CACHE_FILE}")
    except Exception as e:
        log.error(f"Failed to save flight cache: {e}")


def _load_cache(comp_start: datetime, settlement: datetime) -> dict | None:
    """Load flight data from disk cache. Returns None if not available or window mismatch."""
    if not os.path.exists(FLIGHT_CACHE_FILE):
        return None
    try:
        with open(FLIGHT_CACHE_FILE, "r") as f:
            data = json.load(f)
        # Validate cache window matches current config
        cached_start = data.get("_cache_window_start", "")
        cached_end = data.get("_cache_window_end", "")
        if cached_start != comp_start.isoformat() or cached_end != settlement.isoformat():
            log.warning(
                f"Flight cache window mismatch: cached={cached_start}..{cached_end}, "
                f"expected={comp_start.isoformat()}..{settlement.isoformat()}. Re-fetching."
            )
            return None
        dep_count = data.get("dep_count", 0)
        arr_count = data.get("arr_count", 0)
        if dep_count + arr_count > 0:
            log.info(
                f"Loaded flight cache: {dep_count} deps + {arr_count} arrs "
                f"= {dep_count + arr_count} total"
            )
            return data
    except Exception as e:
        log.warning(f"Flight cache corrupted, will re-fetch: {e}")
    return None


# --- Main integration functions ---

def fetch_aerodatabox_flights(
    api_key: str,
    comp_start: datetime,
    settlement: datetime,
) -> dict:
    """Fetch all flight data covering the competition window.

    Uses disk cache if available (flight schedules are static).
    Otherwise fetches from AeroDataBox in 2 calls (12h max each)
    and saves to cache.

    Returns dict with:
      - departures: {flight_id: iso_datetime_str, ...}
      - arrivals: {flight_id: iso_datetime_str, ...}
      - dep_count: int
      - arr_count: int
    """
    # Try cache first
    cached = _load_cache(comp_start, settlement)
    if cached is not None:
        return cached

    if not api_key:
        log.error("No AeroDataBox API key and no cache file")
        return _fallback_data()

    # Convert to local London time strings for the API
    # AeroDataBox expects local time format: YYYY-MM-DDTHH:MM
    london_tz = timezone.utc  # Close enough for UTC-based comp times

    # Split into 2 calls of max 12h each
    midpoint = comp_start + (settlement - comp_start) / 2

    from_1 = comp_start.strftime("%Y-%m-%dT%H:%M")
    to_1 = midpoint.strftime("%Y-%m-%dT%H:%M")
    from_2 = midpoint.strftime("%Y-%m-%dT%H:%M")
    to_2 = settlement.strftime("%Y-%m-%dT%H:%M")

    all_departures: dict[str, str] = {}
    all_arrivals: dict[str, str] = {}

    try:
        # Call 1: first half
        log.info(f"AeroDataBox call 1: {from_1} to {to_1}")
        data1 = _fetch_flights_range(api_key, from_1, to_1)
        _parse_aerodatabox_response(data1, all_departures, all_arrivals)

        # Pause between calls to avoid RapidAPI rate limiting
        time.sleep(3)

        # Call 2: second half
        log.info(f"AeroDataBox call 2: {from_2} to {to_2}")
        data2 = _fetch_flights_range(api_key, from_2, to_2)
        _parse_aerodatabox_response(data2, all_departures, all_arrivals)

    except Exception as e:
        log.error(f"AeroDataBox fetch failed: {e}")
        return _fallback_data()

    result = {
        "departures": all_departures,
        "arrivals": all_arrivals,
        "dep_count": len(all_departures),
        "arr_count": len(all_arrivals),
    }

    dep_count = result["dep_count"]
    arr_count = result["arr_count"]
    log.info(
        f"AeroDataBox flights: {dep_count} deps + {arr_count} arrs "
        f"= {dep_count + arr_count} total"
    )

    # Save to cache for future polls
    _save_cache(result, comp_start, settlement)

    return result


def _parse_aerodatabox_response(
    data: dict,
    departures: dict[str, str],
    arrivals: dict[str, str],
) -> None:
    """Parse AeroDataBox response and add to departure/arrival dicts.

    AeroDataBox response structure: each flight has "movement.scheduledTime.utc".
    Flight number is used as key for deduplication across overlapping API calls.
    Codeshares are already filtered out via withCodeshared=false API parameter.
    """
    for flight in data.get("departures", []):
        flight_num = flight.get("number", "")
        sched = flight.get("movement", {}).get("scheduledTime", {}).get("utc", "")
        if flight_num and sched:
            departures[flight_num] = sched

    for flight in data.get("arrivals", []):
        flight_num = flight.get("number", "")
        sched = flight.get("movement", {}).get("scheduledTime", {}).get("utc", "")
        if flight_num and sched:
            arrivals[flight_num] = sched


def _fallback_data() -> dict:
    """Fallback when no API key and no cache. Conservative estimate."""
    return {
        "departures": {},
        "arrivals": {},
        "dep_count": 0,
        "arr_count": 0,
    }


# --- Theo computation ---

def estimate_total_lhr_count(
    flight_data: dict,
    comp_start: datetime,
    settlement: datetime,
    now: datetime,
) -> tuple[int, float]:
    """Compute LHR_COUNT from AeroDataBox data.

    Returns (total_count, confidence).
    """
    dep_count = flight_data.get("dep_count", 0)
    arr_count = flight_data.get("arr_count", 0)
    total = dep_count + arr_count

    if total == 0:
        return (1300, 200.0)  # fallback

    departures = flight_data.get("departures", {})
    arrivals = flight_data.get("arrivals", {})

    # Count past vs future flights for cancellation adjustment
    past_count = 0
    future_count = 0

    for sched_str in departures.values():
        try:
            t = datetime.fromisoformat(sched_str).replace(tzinfo=timezone.utc)
            if t <= now:
                past_count += 1
            else:
                future_count += 1
        except (ValueError, TypeError):
            past_count += 1  # assume past if unparseable

    for sched_str in arrivals.values():
        try:
            t = datetime.fromisoformat(sched_str).replace(tzinfo=timezone.utc)
            if t <= now:
                past_count += 1
            else:
                future_count += 1
        except (ValueError, TypeError):
            past_count += 1

    # ~1% cancellation rate for future flights (most are already confirmed)
    theo = int(past_count + future_count * 0.99)

    # Confidence: tight for schedule data, wider when more flights are still future
    conf = max(5.0, future_count * 0.3)

    log.info(
        f"LHR_COUNT: {theo} (past={past_count}, future={future_count}, "
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

    for fid, sched_str in departures.items():
        try:
            t = datetime.fromisoformat(sched_str).replace(tzinfo=timezone.utc)
            idx = int((t - comp_start).total_seconds() / interval_seconds)
            if 0 <= idx < total_intervals:
                dep_bins[idx] += 1
        except (ValueError, TypeError):
            continue

    for fid, sched_str in arrivals.items():
        try:
            t = datetime.fromisoformat(sched_str).replace(tzinfo=timezone.utc)
            idx = int((t - comp_start).total_seconds() / interval_seconds)
            if 0 <= idx < total_intervals:
                arr_bins[idx] += 1
        except (ValueError, TypeError):
            continue

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
