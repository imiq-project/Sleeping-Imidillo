# Parking preparation: every lot's month as an hourly occupancy series.
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from data import cleaning, orion, quantumleap
from data.cleaning import CleanedEntity
from data.timewindow import month_window, parse_month

ENTITY_TYPE = "Parking"
ID_PREFIX = "ParkingSpot:"


@dataclass
class ParkingLot:
    entity_id: str
    name: str
    capacity: int
    capacity_note: str            # where the capacity number comes from
    cleaned: CleanedEntity
    occ_pct: pd.Series            # hourly, local time, NaN where no reading
    usable: bool
    exclude_reason: str | None

    @property
    def short_id(self) -> str:
        return self.entity_id.removeprefix(ID_PREFIX)


def _capacity(total_spots, readings: pd.DataFrame) -> tuple[int, str]:
    observed_max = float(readings["freeSpots"].max()) if "freeSpots" in readings and len(readings) else 0.0
    total = int(total_spots) if isinstance(total_spots, (int, float)) and total_spots > 0 else None
    if total and observed_max <= total:
        return total, "totalSpots from Orion"
    if total:
        return int(observed_max), (f"observed max freeSpots ({observed_max:.0f}) exceeds "
                                   f"totalSpots ({total}); using observed max")
    return int(observed_max), "no totalSpots in Orion; using observed max freeSpots"


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


def prepare_parking(start: datetime, end: datetime) -> dict[str, ParkingLot]:
    """All lots for the window [start, end), keyed by short id (e.g. 'Ulrichshaus')."""
    try:
        meta = {e["id"]: e for e in orion.list_entities(ENTITY_TYPE)}
    except orion.OrionError as e:
        print(f"[PARKING] WARNING: Orion unavailable ({e}); capacities from observed maxima")
        meta = {}

    if meta:
        entity_ids = sorted(meta)
    else:  # fall back to whatever QuantumLeap knows
        ents = quantumleap.list_entities()
        entity_ids = sorted(ents.query("entityType == @ENTITY_TYPE")["entityId"])

    lots: dict[str, ParkingLot] = {}
    for entity_id in entity_ids:
        raw = quantumleap.fetch_history(entity_id, start, end)
        cleaned = cleaning.clean(entity_id, raw, start, end)
        info = meta.get(entity_id, {})
        capacity, note = _capacity(info.get("totalSpots"), cleaned.readings)
        usable, reason = _verdict(cleaned)
        lot = ParkingLot(
            entity_id=entity_id,
            name=info.get("name") or entity_id.removeprefix(ID_PREFIX),
            capacity=capacity,
            capacity_note=note,
            cleaned=cleaned,
            occ_pct=_occupancy_pct(cleaned.hourly, capacity),
            usable=usable,
            exclude_reason=reason,
        )
        lots[lot.short_id] = lot
    return lots

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m data.parking <YYYY-MM>")
        return 1
    year, month = parse_month(sys.argv[1])
    start, end = month_window(year, month)
    lots = prepare_parking(start, end)

    print(f"\n{'lot':26s} {'cap':>5s} {'raw':>6s} {'rows':>5s} {'cover':>6s} {'mean%':>6s} {'max%':>5s}  verdict")
    for short_id, lot in lots.items():
        cov = lot.cleaned.coverage
        mean = lot.occ_pct.mean()
        mx = lot.occ_pct.max()
        verdict = "usable" if lot.usable else f"EXCLUDED: {lot.exclude_reason}"
        print(f"{short_id:26s} {lot.capacity:5d} {lot.cleaned.raw_rows:6d} {lot.cleaned.rows:5d} "
              f"{cov['coverage_pct']:5.1f}% {mean if pd.notna(mean) else 0:6.1f} {mx if pd.notna(mx) else 0:5.1f}  {verdict}")
    print("\ncapacity notes:")
    for short_id, lot in lots.items():
        if "totalSpots from Orion" not in lot.capacity_note:
            print(f"  {short_id}: {lot.capacity_note}")
    usable = [s for s, l in lots.items() if l.usable]
    print(f"\n{len(usable)} of {len(lots)} lots usable: {', '.join(usable)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())