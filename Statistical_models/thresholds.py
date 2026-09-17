"""
Threshold model: how often did a pollutant exceed a health limit this month?

The rhythm model says WHEN a pollutant is high; this says whether "high" ever
meant "above the limit". Limits are the EU Air Quality Directive values and
the stricter WHO 2021 guidelines. Counting is deterministic:

  hourly       hours whose value is above the limit
  daily_mean   days whose mean (>= MIN_HOURS_PER_DAY hourly values) is above
  max_8h_mean  days whose highest 8-hour rolling mean is above

INPUT   an hourly series in local time (NaN where no reading)
OUTPUT  one dict per limit with the count, the basis, and the worst value
"""

from __future__ import annotations

import pandas as pd

MIN_HOURS_PER_DAY = 12    # a daily mean needs at least this many hourly values

LIMITS = {
    "no2": [
        {"name": "EU hourly limit", "value": 200, "basis": "hourly", "allowed_per_year": 18},
        {"name": "WHO daily guideline", "value": 25, "basis": "daily_mean"},
    ],
    "pm10": [
        {"name": "EU daily limit", "value": 50, "basis": "daily_mean", "allowed_per_year": 35},
        {"name": "WHO daily guideline", "value": 45, "basis": "daily_mean"},
    ],
    "pm25": [
        {"name": "WHO daily guideline", "value": 15, "basis": "daily_mean"},
    ],
    "o3": [
        {"name": "EU 8-hour target", "value": 120, "basis": "max_8h_mean", "allowed_per_year": 25},
        {"name": "WHO 8-hour guideline", "value": 100, "basis": "max_8h_mean"},
    ],
}


def _daily(series: pd.Series, how: str) -> pd.Series:
    if how == "daily_mean":
        counts = series.resample("D").count()
        means = series.resample("D").mean()
        return means[counts >= MIN_HOURS_PER_DAY]
    if how == "max_8h_mean":
        rolling = series.rolling(8, min_periods=6).mean()
        return rolling.resample("D").max().dropna()
    raise ValueError(how)


def exceedances(series: pd.Series, pollutant: str) -> list[dict]:
    """Exceedance counts of every limit defined for the pollutant."""
    out = []
    s = series.dropna()
    for limit in LIMITS.get(pollutant, []):
        if limit["basis"] == "hourly":
            values, unit_of_count = s, "hours"
        else:
            values, unit_of_count = _daily(s, limit["basis"]), "days"
        n_over = int((values > limit["value"]).sum()) if len(values) else 0
        worst = values.max() if len(values) else None
        out.append({
            "limit": limit["name"],
            "value": limit["value"],
            "basis": limit["basis"],
            "count_over": n_over,
            "count_unit": unit_of_count,
            "evaluated": int(len(values)),
            "worst": round(float(worst), 1) if worst is not None else None,
            "worst_when": values.idxmax().strftime("%Y-%m-%d %H:%M") if len(values) else None,
            **({"allowed_per_year": limit["allowed_per_year"]} if "allowed_per_year" in limit else {}),
        })
    return out
