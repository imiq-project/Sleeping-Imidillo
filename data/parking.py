"""
Parking preparation: every lot's month as an hourly occupancy series.

  1. Orion       -> which lots exist, their name and totalSpots
  2. QuantumLeap -> the month's raw freeSpots rows per lot
  3. cleaning    -> dedupe, local time, hourly grid, coverage
  4. here        -> occupancy % and a usable / excluded verdict, packed as a
                    RhythmSeries for Statistical_models/rhythm.py

Capacity rule. totalSpots in Orion is the capacity UNLESS the sensor reported
more free spots than that during the month (Friedensplatz: totalSpots 90,
freeSpots up to 114). Then the observed maximum is used and the mismatch is
recorded as a note the report can quote.

Exclusion rules (a lot stays in the output with its reason, never silently
dropped): no rows; coverage below cleaning.MIN_COVERAGE_PCT; a flat line.

Run manually:
    python -m data.parking 2026-08
"""

from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd

from data import cleaning, orion, quantumleap
from data.cleaning import CleanedEntity
from data.timewindow import month_window, parse_month
from Statistical_models.rhythm import RhythmSeries

ENTITY_TYPE = "Parking"
ID_PREFIX = "ParkingSpot:"
VARIABLE = "occupancy"
UNIT = "%"
DELTA_UNIT = "percentage points"


def _capacity(total_spots, readings: pd.DataFrame) -> tuple[int, str | None]:
    observed_max = float(readings["freeSpots"].max()) if "freeSpots" in readings and len(readings) else 0.0
    total = int(total_spots) if isinstance(total_spots, (int, float)) and total_spots > 0 else None
    if total and observed_max <= total:
        return total, None
    if total:
        return int(observed_max), (f"observed max freeSpots ({observed_max:.0f}) exceeds totalSpots "
                                   f"({total}); using observed max as capacity")
    return int(observed_max), "no totalSpots in Orion; using observed max freeSpots as capacity"


def _occupancy_pct(hourly: pd.DataFrame, capacity: int) -> pd.Series:
    if "freeSpots" not in hourly or capacity <= 0:
        return pd.Series(dtype=float, index=hourly.index, name="occ_pct")
    occupied = (capacity - hourly["freeSpots"]).clip(lower=0, upper=capacity)
    return (100.0 * occupied / capacity).rename("occ_pct")


def _verdict(cleaned: CleanedEntity) -> tuple[bool, str | None]:
    if cleaned.empty:
        return False, "no data in this window"
    if cleaned.coverage["low_coverage"]:
        return False, f"coverage {cleaned.coverage['coverage_pct']}% is below {cleaning.MIN_COVERAGE_PCT:.0f}%"
    if "freeSpots" not in cleaned.readings:
        return False, "no freeSpots attribute"
    if cleaned.readings["freeSpots"].nunique() <= 1:
        return False, "flat line: one value all month, sensor not updating"
    return True, None


def prepare_parking(start: datetime, end: datetime) -> dict[str, RhythmSeries]:
    """All lots for [start, end), keyed by short id (e.g. 'Ulrichshaus')."""
    try:
        meta = {e["id"]: e for e in orion.list_entities(ENTITY_TYPE)}
    except orion.OrionError as e:
        print(f"[PARKING] WARNING: Orion unavailable ({e}); capacities from observed maxima")
        meta = {}
    if meta:
        entity_ids = sorted(meta)
    else:
        ents = quantumleap.list_entities()
        entity_ids = sorted(ents.query("entityType == @ENTITY_TYPE")["entityId"])

    out: dict[str, RhythmSeries] = {}
    for entity_id in entity_ids:
        raw = quantumleap.fetch_history(entity_id, start, end)
        cleaned = cleaning.clean(entity_id, raw, start, end)
        info = meta.get(entity_id, {})
        capacity, note = _capacity(info.get("totalSpots"), cleaned.readings)
        usable, reason = _verdict(cleaned)
        key = entity_id.removeprefix(ID_PREFIX)
        out[key] = RhythmSeries(
            key=key,
            name=info.get("name") or key,
            values=_occupancy_pct(cleaned.hourly, capacity),
            coverage=cleaned.coverage,
            usable=usable,
            exclude_reason=reason,
            meta={"capacity": capacity, "note": note, "raw_rows": cleaned.raw_rows, "rows": cleaned.rows},
        )
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m data.parking <YYYY-MM>")
        return 1
    start, end = month_window(*parse_month(sys.argv[1]))
    lots = prepare_parking(start, end)
    print(f"\n{'lot':26s} {'cap':>5s} {'raw':>6s} {'rows':>5s} {'cover':>6s} {'mean%':>6s}  verdict")
    for key, s in lots.items():
        mean = s.values.mean()
        print(f"{key:26s} {s.meta['capacity']:5d} {s.meta['raw_rows']:6d} {s.meta['rows']:5d} "
              f"{s.coverage['coverage_pct']:5.1f}% {mean if pd.notna(mean) else 0:6.1f}  "
              f"{'usable' if s.usable else 'EXCLUDED: ' + s.exclude_reason}")
    for key, s in lots.items():
        if s.meta.get("note"):
            print(f"  note {key}: {s.meta['note']}")
    print(f"\n{sum(s.usable for s in lots.values())} of {len(lots)} lots usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
