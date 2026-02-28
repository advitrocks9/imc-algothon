"""Weather data fetching from Open-Meteo (free, no key required)."""
import requests
import pandas as pd
import logging

log = logging.getLogger("data.weather")

LONDON_LAT, LONDON_LON = 51.5074, -0.1278


def c_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


def fetch_weather(past_steps: int = 96, forecast_steps: int = 96) -> pd.DataFrame:
    """Fetch 15-min weather: past + forecast. Adds temp_f column.

    96 steps = 24 hours.
    """
    variables = "temperature_2m,relative_humidity_2m"
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
        timeout=15,
    )
    resp.raise_for_status()
    m = resp.json()["minutely_15"]
    df = pd.DataFrame({
        "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
        "temperature_c": m["temperature_2m"],
        "humidity": m["relative_humidity_2m"],
    })
    df["temperature_f"] = df["temperature_c"].apply(c_to_f)
    log.info(f"Weather fetched: {len(df)} rows")
    return df
