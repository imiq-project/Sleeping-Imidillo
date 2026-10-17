"""
Weather model: what the month's weather was, and how it compares with the
same days in earlier years.

The temperature's daily cycle and its month-on-month change go through the
rhythm model (all days pooled, HAC_LAGS: warm and cold spells last days, so
one day of lags would overstate certainty). This module adds what the rhythm
model cannot say:

SUMMARY (deterministic counts, DWD climate-day definitions)
  temperature    mean, lowest and highest hourly value, with when
  summer days    daily maximum >= 25 °C        hot days   maximum >= 30 °C
  frost days     daily minimum <  0 °C         ice days   maximum <  0 °C
  precipitation  total, wet days (>= 1.0 mm), wettest day
  wind           strongest gust, mean wind speed
  A day counts only with at least MIN_HOURS_PER_DAY hourly values. Daily
  maxima and minima come from hourly values, so they can be slightly below
  the DWD's official 10-minute extremes.

CLIMATE (the same calendar days in every earlier year since 2010)
  For a running month the window is the days covered so far (1 to 26
  October against 1 to 26 October of 2010-2025), so early-month warmth is
  not compared with whole months. A year counts with >= MIN_YEAR_COVERAGE_PCT
  of its hours. "Usual range" = reference mean +- one standard deviation of
  the reference years; outside_usual_range says whether the period left it.
  The rank counts the current period among all years used (1 = warmest /
  wettest).

INPUT   data.weather.WeatherPeriod objects
OUTPUT  JSON-serialisable dicts, numbers rounded
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
from data.weather import WeatherPeriod

HAC_LAGS = 24 * 7             # one week of hourly lags for the temperature model
MIN_HOURS_PER_DAY = 20        # hourly values a day needs to count in the day statistics
MIN_YEAR_COVERAGE_PCT = 90.0  # a reference year needs this much of its hours
WET_DAY_MM = 1.0


def _r(x, decimals: int = 1):
    return None if x is None or pd.isna(x) else round(float(x), decimals)


def _when(series: pd.Series, how: str) -> dict:
    s = series.dropna()
    if s.empty:
        return {"value": None, "when": None}
    ts = s.idxmax() if how == "max" else s.idxmin()
    return {"value": _r(s[ts]), "when": ts.strftime("%Y-%m-%d %H:%M")}


def _days(series: pd.Series, how: str) -> pd.Series:
    """Daily max/min/sum on days with at least MIN_HOURS_PER_DAY values."""
    counts = series.resample("D").count()
    agg = getattr(series.resample("D"), how)()
    return agg[counts >= MIN_HOURS_PER_DAY]


def summary(p: WeatherPeriod) -> dict:
    h = p.hourly
    t, rain = h["temperature"], h["precipitation"]
    tmax, tmin = _days(t, "max"), _days(t, "min")
    rain_days = _days(rain, "sum")
    wettest = {"date": rain_days.idxmax().strftime("%Y-%m-%d"), "mm": _r(rain_days.max())} \
        if not rain_days.empty and rain_days.max() > 0 else None
    return {
        "hours": int(len(h)),
        "temperature": {
            "mean": _r(t.mean()),
            "lowest": _when(t, "min"),
            "highest": _when(t, "max"),
            "days_evaluated": int(len(tmax)),
            "summer_days_25": int((tmax >= 25).sum()),
            "hot_days_30": int((tmax >= 30).sum()),
            "frost_days": int((tmin < 0).sum()),
            "ice_days": int((tmax < 0).sum()),
        },
        "precipitation": {
            "total_mm": _r(rain.sum(min_count=1)),
            "hours_with_value": int(rain.notna().sum()),
            "days_evaluated": int(len(rain_days)),
            "wet_days_1mm": int((rain_days >= WET_DAY_MM).sum()),
            "wettest_day": wettest,
        },
        "wind": {
            "strongest_gust_kmh": _when(h["wind_gust_speed"], "max"),
            "mean_speed_kmh": _r(h["wind_speed"].mean()),
        },
    }


def daily(p: WeatherPeriod) -> list[dict]:
    """One row per day for the chart (dropped from the LLM copy)."""
    t, rain = p.hourly["temperature"], p.hourly["precipitation"]
    frame = pd.DataFrame({"mean": t.resample("D").mean(), "min": t.resample("D").min(),
                          "max": t.resample("D").max(), "precipitation": rain.resample("D").sum(min_count=1)})
    return [{"date": d.strftime("%Y-%m-%d"), **{k: _r(v) for k, v in row.items()}} for d, row in frame.iterrows()]


def _window_label(p: WeatherPeriod) -> str:
    tz = ZoneInfo(config.LOCAL_TZ)
    first = p.start.astimezone(tz)
    last = (p.end - pd.Timedelta(hours=1)).astimezone(tz)
    if first.month == last.month:
        return f"{first.day} to {last.day} {first.strftime('%B')}"
    return f"{first.day} {first.strftime('%B')} to {last.day} {last.strftime('%B')}"


def _compare(current: float | None, by_year: dict[int, float], higher_word: str) -> dict | None:
    if current is None or len(by_year) < 5:
        return None
    ref = np.array(list(by_year.values()), dtype=float)
    mean, sd = float(ref.mean()), float(ref.std(ddof=1))
    values = sorted([*by_year.values(), current], reverse=True)
    top = max(by_year, key=by_year.get)
    bottom = min(by_year, key=by_year.get)
    return {
        "current": _r(current),
        "reference_mean": _r(mean),
        "reference_sd": _r(sd),
        "anomaly": _r(current - mean),
        "z": _r((current - mean) / sd if sd > 0 else 0.0),
        "outside_usual_range": bool(abs(current - mean) > sd),
        f"rank_{higher_word}": values.index(current) + 1,
        "n_years_ranked": len(values),
        "highest_reference_year": {"year": top, "value": _r(by_year[top])},
        "lowest_reference_year": {"year": bottom, "value": _r(by_year[bottom])},
    }


def climate(current: WeatherPeriod, past: dict[int, WeatherPeriod]) -> dict:
    used, skipped, temps, rains = [], {}, {}, {}
    for year, p in sorted(past.items()):
        if p.coverage["coverage_pct"] < MIN_YEAR_COVERAGE_PCT:
            skipped[year] = f"temperature coverage {p.coverage['coverage_pct']}%"
            continue
        used.append(year)
        temps[year] = float(p.hourly["temperature"].mean())
        rain = p.hourly["precipitation"]
        if 100.0 * rain.notna().mean() >= MIN_YEAR_COVERAGE_PCT:
            rains[year] = float(rain.sum())
    cur_t = current.hourly["temperature"].mean()
    cur_rain = current.hourly["precipitation"].sum(min_count=1)
    return {
        "window": _window_label(current),
        "reference_years": f"{used[0]}-{used[-1]}" if used else None,
        "years_used": used,
        "years_skipped": skipped,
        "temperature_mean": _compare(None if pd.isna(cur_t) else float(cur_t), temps, "warmest"),
        "precipitation_total_mm": _compare(None if pd.isna(cur_rain) else float(cur_rain), rains, "wettest"),
        "by_year": [{"year": y, "temperature_mean": _r(temps[y]), "precipitation_total_mm": _r(rains.get(y))}
                    for y in used],
    }
