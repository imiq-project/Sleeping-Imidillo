"""
Weather preparation: hourly measurements of the DWD station Magdeburg.

Not the digital twin's data. The twin's own weather sensors are too sparse
(probed 2026-10-01: one of nine usable), so the report takes the weather
from the Deutscher Wetterdienst (DWD), through the free Bright Sky API
(https://brightsky.dev, no key). The report must name the source:
"Source: Deutscher Wetterdienst".

Probed 2026-10-17:
  - station 03126 "Magdeburg", hourly since 2010, about 3.6 km from the city centre
  - asked by dwd_station_id, Bright Sky does NOT fill gaps from neighbouring
    stations, so a missing hour stays missing (asked by lat/lon it would)
  - observation_type "historical" = quality-checked hourly data (to about
    two days ago), "current" = the latest raw observations, "forecast" =
    MOSMIX forecast for hours not yet observed. Forecast rows are dropped:
    only measurements go into the report
  - no sunshine or solar values at this station
  - last_date is inclusive, so rows are cut to [start, end) here

Units (Bright Sky "dwd" units): temperature °C, precipitation mm in the
hour, wind and gust speed km/h, relative humidity %.

Run manually:
    python -m data.weather 2026-09
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config
from data import cleaning
from data.timewindow import month_window, parse_month
from Statistical_models.rhythm import RhythmSeries

STATION_ID = "03126"
STATION_NAME = "Magdeburg"
SOURCE = "Deutscher Wetterdienst (DWD), station Magdeburg, via Bright Sky"
ATTRIBUTION = "Source: Deutscher Wetterdienst"
FIELDS = ["temperature", "precipitation", "wind_speed", "wind_gust_speed", "relative_humidity"]
VARIABLE = "air temperature"
UNIT = "°C"
DELTA_UNIT = "°C"
CLIMATE_FIRST_YEAR = 2010      # first year of hourly data at the station
_RETRIES = 3
_BACKOFF_S = 2.0


class WeatherError(RuntimeError):
    """Bright Sky unreachable or returned something we cannot interpret."""


@dataclass
class WeatherPeriod:
    start: datetime
    end: datetime
    hourly: pd.DataFrame                          # FIELDS on the full hourly grid, local time, NaN = missing
    coverage: dict                                # temperature coverage, as for every sensor
    observation_hours: dict = field(default_factory=dict)   # {"historical": n, "current": n}


def _get(params: dict) -> dict:
    url = f"{config.BRIGHTSKY_BASE_URL.rstrip('/')}/weather"
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers={"Accept": "application/json"},
                                timeout=config.BRIGHTSKY_TIMEOUT_S)
        except requests.RequestException as e:
            last_error = e
        else:
            if resp.status_code < 500:
                if resp.status_code != 200:
                    raise WeatherError(f"GET {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
                try:
                    return resp.json()
                except ValueError as e:
                    raise WeatherError(f"GET {url} returned no JSON: {e}") from e
            last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if attempt < _RETRIES:
            time.sleep(_BACKOFF_S * attempt)
    raise WeatherError(f"GET {url} failed after {_RETRIES} attempts: {last_error}")


def fetch_period(start: datetime, end: datetime) -> WeatherPeriod:
    """Measured hourly weather at the station for [start, end), tz-aware datetimes."""
    payload = _get({"dwd_station_id": STATION_ID, "date": start.isoformat(), "last_date": end.isoformat()})
    kinds = {s["id"]: s.get("observation_type") for s in payload.get("sources", [])}
    rows = [r for r in payload.get("weather", []) if kinds.get(r.get("source_id")) in ("historical", "current")]
    tz = config.LOCAL_TZ
    grid = pd.date_range(start, end, freq="1h", inclusive="left").tz_convert(tz)
    if rows:
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
        df["kind"] = df["source_id"].map(kinds)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df[(df.index >= grid[0]) & (df.index < pd.Timestamp(end).tz_convert(tz))] if len(grid) else df.iloc[0:0]
        kind_counts = df["kind"].value_counts().to_dict()
        hourly = df.reindex(columns=FIELDS).apply(pd.to_numeric, errors="coerce").reindex(grid)
    else:
        kind_counts = {}
        hourly = pd.DataFrame(index=grid, columns=FIELDS, dtype=float)
    temps = hourly[["temperature"]]
    readings = temps.dropna()
    return WeatherPeriod(start=start, end=end, hourly=hourly,
                         coverage=cleaning.coverage(temps, readings),
                         observation_hours={k: int(kind_counts.get(k, 0)) for k in ("historical", "current")})


def temperature_series(period: WeatherPeriod) -> dict[str, RhythmSeries]:
    """The temperature as a one-series family for the rhythm model."""
    usable = not period.coverage["low_coverage"]
    reason = None if usable else (f"coverage {period.coverage['coverage_pct']}% is below "
                                  f"{cleaning.MIN_COVERAGE_PCT:.0f}%")
    return {STATION_NAME: RhythmSeries(
        key=STATION_NAME, name=f"DWD station {STATION_NAME}", values=period.hourly["temperature"],
        coverage=period.coverage, usable=usable, exclude_reason=reason,
        meta={"dwd_station_id": STATION_ID})}


def same_window_in_year(start: datetime, end: datetime, year: int) -> tuple[datetime, datetime]:
    """[start, end) moved to another year at the same local wall-clock dates
    (1 to 26 October 2026 -> 1 to 26 October 2015). Feb 29 becomes Feb 28."""
    tz = ZoneInfo(config.LOCAL_TZ)
    s_local, e_local = start.astimezone(tz), end.astimezone(tz)
    length = e_local.replace(tzinfo=None) - s_local.replace(tzinfo=None)
    day = min(s_local.day, 28) if s_local.month == 2 else s_local.day
    s_new = s_local.replace(tzinfo=None, year=year, day=day)
    return s_new.replace(tzinfo=tz), (s_new + length).replace(tzinfo=tz)


def fetch_past_years(start: datetime, end: datetime, first_year: int = CLIMATE_FIRST_YEAR) -> dict[int, WeatherPeriod]:
    """The same window in every earlier year from first_year, for the climate comparison."""
    year = start.astimezone(ZoneInfo(config.LOCAL_TZ)).year
    return {y: fetch_period(*same_window_in_year(start, end, y)) for y in range(first_year, year)}


# --------------------------------------------------------------------------- #
# Manual check                                                                 #
# --------------------------------------------------------------------------- #

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m data.weather <YYYY-MM>")
        return 1
    start, end = month_window(*parse_month(sys.argv[1]))
    p = fetch_period(start, end)
    h = p.hourly
    print(f"[WEATHER] {SOURCE}")
    print(f"[WEATHER] {len(h)} hours, temperature coverage {p.coverage['coverage_pct']}%, "
          f"longest gap {p.coverage['longest_gap_h']} h; observation hours {p.observation_hours}")
    print(f"[WEATHER] temperature {h['temperature'].min():.1f} .. {h['temperature'].max():.1f} °C, "
          f"mean {h['temperature'].mean():.1f}; precipitation {h['precipitation'].sum():.1f} mm; "
          f"max gust {h['wind_gust_speed'].max():.0f} km/h")
    past = fetch_past_years(start, end, first_year=end.year - 3)
    for y, pp in past.items():
        print(f"[WEATHER] {y}: mean {pp.hourly['temperature'].mean():.1f} °C, "
              f"precipitation {pp.hourly['precipitation'].sum():.1f} mm, coverage {pp.coverage['coverage_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
