"""
Air-quality preparation: every (pollutant, station) as an hourly series.

Five AirQuality stations, each measuring a different subset of NO2, O3, PM10
and PM2.5 (µg/m³); the other attributes (pressure, CO2, humidity,
temperature) are weather and not modelled here. Output is one RhythmSeries
per station under each pollutant, so the rhythm model runs once per
pollutant with the stations as its series.

Staleness (probed 2026-09-17: PM values unchanged in ~75% of hours at West):
a series whose value is identical to the previous hour in more than
STALE_MAX_FRACTION of the hours is excluded as "sensor updates too rarely".
The rhythm model would otherwise report a flat, meaningless daily profile.

Run manually:
    python -m data.air_quality 2026-08
"""

from __future__ import annotations

import re
import sys
from datetime import datetime

import pandas as pd

from data import cleaning, orion, quantumleap
from data.timewindow import month_window, parse_month
from Statistical_models.rhythm import RhythmSeries

ENTITY_TYPE = "AirQuality"
ID_PREFIX = "Air:"
POLLUTANTS = {          # attribute -> (label, unit)
    "no2": ("NO2", "µg/m³"),
    "o3": ("O3", "µg/m³"),
    "pm10": ("PM10", "µg/m³"),
    "pm25": ("PM2.5", "µg/m³"),
}
VARIABLE = "concentration"
DELTA_UNIT = "µg/m³"
STALE_MAX_FRACTION = 0.5


def _verdict(values: pd.Series, coverage: dict) -> tuple[bool, str | None, float]:
    stale = float((values.diff() == 0).mean()) if values.notna().sum() > 1 else 0.0
    if coverage["hours_with_data"] == 0:
        return False, "no data in this window", stale
    if coverage["low_coverage"]:
        return False, f"coverage {coverage['coverage_pct']}% is below {cleaning.MIN_COVERAGE_PCT:.0f}%", stale
    if values.dropna().nunique() <= 1:
        return False, "flat line: one value all month, sensor not updating", stale
    if stale > STALE_MAX_FRACTION:
        return False, f"stale: value unchanged from the previous hour in {stale * 100:.0f}% of hours", stale
    return True, None, stale


def prepare_air_quality(start: datetime, end: datetime) -> dict[str, dict[str, RhythmSeries]]:
    """{pollutant: {station: RhythmSeries}} for [start, end)."""
    try:
        station_ids = sorted(e["id"] for e in orion.list_entities(ENTITY_TYPE))
    except orion.OrionError as e:
        print(f"[AIR] WARNING: Orion unavailable ({e}); using QuantumLeap inventory")
        ents = quantumleap.list_entities()
        station_ids = sorted(ents.query("entityType == @ENTITY_TYPE")["entityId"])

    out: dict[str, dict[str, RhythmSeries]] = {p: {} for p in POLLUTANTS}
    for entity_id in station_ids:
        raw = quantumleap.fetch_history(entity_id, start, end)
        cleaned = cleaning.clean(entity_id, raw, start, end)
        station = entity_id.removeprefix(ID_PREFIX)
        for attr, (label, unit) in POLLUTANTS.items():
            if attr not in cleaned.hourly.columns:
                continue                      # this station does not measure it at all
            values = cleaned.hourly[attr]
            readings = cleaned.readings[[attr]].dropna() if attr in cleaned.readings else cleaned.readings.iloc[0:0]
            cov = cleaning.coverage(cleaned.hourly[[attr]], readings)
            if cov["hours_with_data"] == 0 and readings.empty:
                continue                      # attribute exists in the schema but never reported here
            usable, reason, stale = _verdict(values, cov)
            out[attr][station] = RhythmSeries(
                key=station, name=re.sub(r"(?<=[a-z])(?=[A-Z])", " ", station), values=values, coverage=cov,
                usable=usable, exclude_reason=reason,
                meta={"pollutant": label, "unit": unit, "stale_fraction": round(stale, 2),
                      "raw_rows": cleaned.raw_rows, "rows": int(len(readings))},
            )
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m data.air_quality <YYYY-MM>")
        return 1
    start, end = month_window(*parse_month(sys.argv[1]))
    fam = prepare_air_quality(start, end)
    for attr, stations in fam.items():
        label, unit = POLLUTANTS[attr]
        print(f"\n=== {label} ({unit}) ===")
        for key, s in stations.items():
            mean = s.values.mean()
            print(f"  {key:18s} rows {s.meta['rows']:5d}  cover {s.coverage['coverage_pct']:5.1f}%  "
                  f"stale {s.meta['stale_fraction']:.2f}  mean {mean if pd.notna(mean) else 0:6.1f}  "
                  f"{'usable' if s.usable else 'EXCLUDED: ' + s.exclude_reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
