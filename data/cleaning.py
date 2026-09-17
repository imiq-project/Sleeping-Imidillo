# Cleaning: raw QuantumLeap rows to tidy hourly series + coverage facts.

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

import config
from data.quantumleap import fetch_history
from data.timewindow import month_window, parse_month

MIN_COVERAGE_PCT = 60.0   # below this the entity is flagged "sensor quiet"; raise back to 70 once data quality improves


@dataclass
class CleanedEntity:
    entity_id: str
    raw_rows: int
    rows: int
    readings: pd.DataFrame
    hourly: pd.DataFrame
    coverage: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.rows == 0


# --------------------------------------------------------------------------- #
# Steps                                                                        #
# --------------------------------------------------------------------------- #

def dedupe(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse burst-logged duplicates: same minute + same values -> one row."""
    if raw.empty:
        return raw
    df = raw.sort_index()
    minute = df.index.floor("min")
    keep = ~pd.DataFrame(df.values, index=df.index).assign(_minute=minute.values) \
        .duplicated(keep="first").values
    return df[keep]


def to_local(df: pd.DataFrame) -> pd.DataFrame:
    """UTC index -> report time zone. Every hour/weekday downstream is local."""
    if df.empty:
        return df
    return df.tz_convert(config.LOCAL_TZ)


def hourly_grid(readings: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly mean of numeric columns on the full [start, end) grid, local time.
    Hours without any reading are NaN, not dropped."""
    grid = pd.date_range(
        start.astimezone(readings.index.tz) if not readings.empty else start,
        end.astimezone(readings.index.tz) if not readings.empty else end,
        freq="1h", inclusive="left", name="timestamp",
    ).tz_convert(config.LOCAL_TZ)
    numeric = readings.select_dtypes("number") if not readings.empty else readings
    if numeric.empty:
        return pd.DataFrame(index=grid)
    return numeric.resample("1h").mean().reindex(grid)


def coverage(hourly: pd.DataFrame, readings: pd.DataFrame) -> dict:
    """How much of the window the sensor actually reported."""
    hours_in_window = int(len(hourly))
    has_data = hourly.notna().any(axis=1) if not hourly.empty else pd.Series(dtype=bool)
    hours_with_data = int(has_data.sum())
    pct = 100.0 * hours_with_data / hours_in_window if hours_in_window else 0.0

    longest_gap_h, gap_start = 0, None
    if hours_in_window and not has_data.all():
        missing = ~has_data
        run_id = (missing != missing.shift()).cumsum()
        gap_len = missing.groupby(run_id).sum()
        worst = gap_len.idxmax()
        longest_gap_h = int(gap_len.max())
        gap_start = hourly.index[(run_id == worst).to_numpy()][0].strftime("%Y-%m-%d %H:%M")

    return {
        "hours_in_window": hours_in_window,
        "hours_with_data": hours_with_data,
        "coverage_pct": round(pct, 1),
        "longest_gap_h": longest_gap_h,
        "longest_gap_start": gap_start,
        "first_reading": readings.index.min().strftime("%Y-%m-%d %H:%M") if len(readings) else None,
        "last_reading": readings.index.max().strftime("%Y-%m-%d %H:%M") if len(readings) else None,
        "low_coverage": bool(pct < MIN_COVERAGE_PCT),
    }


# --------------------------------------------------------------------------- #
# The one function other modules call                                          #
# --------------------------------------------------------------------------- #

def clean(entity_id: str, raw: pd.DataFrame, start: datetime, end: datetime) -> CleanedEntity:
    """Raw rows for one entity -> CleanedEntity for the window [start, end)."""
    readings = to_local(dedupe(raw))
    hourly = hourly_grid(readings, start, end)
    return CleanedEntity(
        entity_id=entity_id,
        raw_rows=int(len(raw)),
        rows=int(len(readings)),
        readings=readings,
        hourly=hourly,
        coverage=coverage(hourly, readings),
    )


# --------------------------------------------------------------------------- #
# Manual check                                                                 #
# --------------------------------------------------------------------------- #

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m data.cleaning <entityId> <YYYY-MM>")
        return 1
    entity_id, (year, month) = sys.argv[1], parse_month(sys.argv[2])
    start, end = month_window(year, month)

    raw = fetch_history(entity_id, start, end)
    ce = clean(entity_id, raw, start, end)

    print(f"[CLEAN] {entity_id}: {ce.raw_rows} raw rows -> {ce.rows} after dedupe")
    if ce.empty:
        print("[CLEAN] nothing in this window")
        return 0
    cov = ce.coverage
    print(f"[CLEAN] readings {cov['first_reading']} .. {cov['last_reading']} ({config.LOCAL_TZ})")
    print(f"[CLEAN] coverage {cov['coverage_pct']}% of {cov['hours_in_window']} hours"
          f"{'  LOW' if cov['low_coverage'] else ''}")
    if cov["longest_gap_h"]:
        print(f"[CLEAN] longest gap {cov['longest_gap_h']} h from {cov['longest_gap_start']}")
    print(f"[CLEAN] hourly columns: {list(ce.hourly.columns)}")
    print(ce.hourly.head(6).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())