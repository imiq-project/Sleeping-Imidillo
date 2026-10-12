"""
EV charging preparation: every station's month as an hourly occupancy series.

18 public charging stations, each with `capacity` charging points (2 at every
station, probed 2026-10-05) and `availableCapacity` = points free right now.

The data is EVENT-DRIVEN (probed 2026-10-05 on June-September):
  - each change is written as a burst of identical rows within one second;
    after collapsing the bursts, every remaining row differs from the previous
    one, i.e. a row means "a car arrived" or "a car left";
  - an hour without a row is therefore NOT missing data: the state simply did
    not change.
So the state is carried forward from each change to the next (on a 1-minute
grid) and the hourly value is the time-weighted mean of that state. Averaging
only the change rows and dropping empty hours (the first student version)
used ~30% of the hours and overstated occupancy, e.g. AmBlauenBock3 0.93 vs
0.64 occupied points, busiest hour 05:00 vs 13:00.

Because rows only come on changes, "unchanged" and "sensor offline" look the
same. A state is carried for at most MAX_CARRY_H hours; after that the station
counts as silent until its next row.

occupancy % = 100 x (capacity - availableCapacity) / capacity, so with two
points 50 % means one car charging. Per station meta also carries
busy_all_points_pct: share of known time when no point was free.

Credits: duplicate handling and the capacity interpretation follow the student
model (ev_charging_hourly_regression.py, 2026-10).

Run manually:
    python -m data.ev_charging 2026-09
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta

import pandas as pd

import config
from data import cleaning, orion, quantumleap
from data.timewindow import month_window, parse_month
from Statistical_models.rhythm import RhythmSeries

ENTITY_TYPE = "EVChargingStation"
ID_PREFIX = "EVChargingStation:"
ATTR = "availableCapacity"
VARIABLE = "charging point occupancy"
UNIT = "%"
DELTA_UNIT = "percentage points"
DEFAULT_CAPACITY = 2
MAX_CARRY_H = 24          # carry a state at most this long without a new row
MIN_CHANGES = 10          # fewer state changes in the period -> not reporting


def _events(raw: pd.DataFrame) -> pd.Series:
    """Raw rows -> one value per state change, local time.

    cleaning.dedupe collapses identical rows within the same minute, which on
    this data is the same as the student's 1-second rule (no rows at all
    between 1 s and 10 min apart). Rows that repeat the previous value are
    dropped too: they carry no new information about the state."""
    if raw.empty or ATTR not in raw:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=config.LOCAL_TZ))
    s = cleaning.to_local(cleaning.dedupe(raw))[ATTR].dropna()
    s = s[s.isin([0, 1, 2, 3, 4, 5, 6, 7, 8])]            # counts only; anything else is junk
    return s[s.diff().fillna(1) != 0]


def _state_hourly(events: pd.Series, capacity: int, start: datetime, end: datetime) -> pd.Series:
    """Carry each state forward (max MAX_CARRY_H) on a 1-minute grid and
    return the hourly time-weighted occupancy % on the full [start, end) grid."""
    tz = config.LOCAL_TZ
    hours = pd.date_range(start, end, freq="1h", inclusive="left").tz_convert(tz)
    if events.empty:
        return pd.Series(float("nan"), index=hours, name="occ_pct")
    minutes = pd.date_range(start, end, freq="1min", inclusive="left").tz_convert(tz)
    occ = (100.0 * (capacity - events) / capacity).clip(0, 100)
    state = occ.reindex(occ.index.union(minutes)).ffill()
    last_change = pd.Series(occ.index, index=occ.index).reindex(state.index).ffill()
    age = state.index.to_series() - last_change
    state[age > pd.Timedelta(hours=MAX_CARRY_H)] = float("nan")
    state = state.reindex(minutes)
    hourly = state.resample("1h").mean().reindex(hours)
    hourly.attrs["busy_all_pct"] = (float((state >= 99.9).sum() / state.notna().sum() * 100)
                                    if state.notna().any() else None)
    return hourly.rename("occ_pct")


def _verdict(n_changes: int, coverage: dict) -> tuple[bool, str | None]:
    if coverage["hours_with_data"] == 0:
        return False, "no data in this window"
    if n_changes < MIN_CHANGES:
        return False, f"only {n_changes} state changes, station probably not reporting"
    if coverage["low_coverage"]:
        return False, (f"state known for only {coverage['coverage_pct']}% of hours "
                       f"(below {cleaning.MIN_COVERAGE_PCT:.0f}%)")
    return True, None


def prepare_ev_charging(start: datetime, end: datetime) -> dict[str, RhythmSeries]:
    """All charging stations for [start, end), keyed by short id."""
    try:
        meta = {e["id"]: e for e in orion.list_entities(ENTITY_TYPE)}
    except orion.OrionError as e:
        print(f"[EV] WARNING: Orion unavailable ({e}); capacity {DEFAULT_CAPACITY} assumed")
        meta = {}
    ql = quantumleap.list_entities()
    entity_ids = sorted(set(meta) | set(ql.query("entityType == @ENTITY_TYPE")["entityId"]))

    out: dict[str, RhythmSeries] = {}
    lookback = start - timedelta(hours=MAX_CARRY_H)            # to know the state at the window start
    for entity_id in entity_ids:
        info = meta.get(entity_id, {})
        cap = info.get("capacity")
        capacity = int(cap) if isinstance(cap, (int, float)) and cap > 0 else DEFAULT_CAPACITY
        raw = quantumleap.fetch_history(entity_id, lookback, end)
        events = _events(raw)
        hourly = _state_hourly(events, capacity, start, end)
        in_window = events[events.index >= pd.Timestamp(start).tz_convert(config.LOCAL_TZ)]
        coverage = cleaning.coverage(hourly.to_frame(), in_window.to_frame())
        usable, reason = _verdict(len(in_window), coverage)
        key = entity_id.removeprefix(ID_PREFIX)
        busy = hourly.attrs.get("busy_all_pct")
        out[key] = RhythmSeries(
            key=key,
            name=info.get("name") or re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", key),
            values=hourly,
            coverage=coverage,
            usable=usable,
            exclude_reason=reason,
            meta={"capacity": capacity,
                  "busy_all_points_pct": round(busy, 1) if busy is not None else None,
                  "state_changes": int(len(in_window)),
                  "raw_rows": int(len(raw))},
        )
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m data.ev_charging <YYYY-MM>")
        return 1
    start, end = month_window(*parse_month(sys.argv[1]))
    stations = prepare_ev_charging(start, end)
    print(f"\n{'station':28s} {'cap':>3s} {'raw':>5s} {'chg':>4s} {'known':>6s} {'occ%':>5s} {'full%':>6s}  verdict")
    for key, s in stations.items():
        m = s.meta
        mean = s.values.mean()
        full = m["busy_all_points_pct"]
        print(f"{key:28s} {m['capacity']:3d} {m['raw_rows']:5d} {m['state_changes']:4d} "
              f"{s.coverage['coverage_pct']:5.1f}% {mean if pd.notna(mean) else 0:5.1f} "
              f"{full if full is not None else 0:5.1f}%  "
              f"{'usable' if s.usable else 'EXCLUDED: ' + s.exclude_reason}")
    print(f"\n{sum(s.usable for s in stations.values())} of {len(stations)} stations usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
