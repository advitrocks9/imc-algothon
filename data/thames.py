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
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return pd.DataFrame(columns=["time", "level"])
    df = pd.DataFrame(items)[["dateTime", "value"]].rename(
        columns={"dateTime": "time", "value": "level"}
    )
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
    df["level"] = df["level"].astype(float)
    return df.sort_values("time").reset_index(drop=True)


class TidalModel:
    """Fits a harmonic model to observed tidal data and predicts future levels.

    Model: h(t) = H0 + Σ Ai*cos(ωi*t + φi)

    Primary constituents:
    - M2 (principal lunar semidiurnal): period ≈ 12.4206 hours
    - S2 (principal solar semidiurnal): period ≈ 12.0000 hours
    - K1 (luni-solar diurnal): period ≈ 23.9345 hours
    - O1 (principal lunar diurnal): period ≈ 25.8193 hours
    """

    CONSTITUENTS = {
        "M2": 12.4206,
        "S2": 12.0000,
        "K1": 23.9345,
        "O1": 25.8193,
    }

    def __init__(self):
        self.coeffs: np.ndarray | None = None
        self._t0: datetime | None = None
        self.rmse: float = 0.5
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
