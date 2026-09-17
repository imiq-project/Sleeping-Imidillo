"""
Traffic preparation: every street sensor's month as an hourly speed series.

The Traffic entities carry one measured attribute, avgSpeed (km/h), plus a
static speedLimit. Orion lists ~1000 street segments but only the ~29 with a
history in QuantumLeap are sensors; the QuantumLeap inventory is the list.

Plausibility (probed 2026-09-17: readings up to 185 km/h on a 50 km/h street):
  - avgSpeed <= 0            -> no vehicle measured, treated as no reading
  - avgSpeed > MAX_PLAUSIBLE  -> sensor fault, dropped
Both counts are kept in meta so the report can mention them.

Average speed is a weak proxy for traffic (it mostly reflects the speed
limit); it is what the sensors provide. If a count / intensity attribute
appears later, add it here as a second series family.

Run manually:
    python -m data.traffic 2026-08
"""

from __future__ import annotations

import re
import sys
from datetime import datetime

import pandas as pd

from data import cleaning, orion, quantumleap
from data.cleaning import CleanedEntity
from data.timewindow import month_window, parse_month
from Statistical_models.rhythm import RhythmSeries

ENTITY_TYPE = "Traffic"
ID_PREFIX = "Traffic:"
ATTR = "avgSpeed"
VARIABLE = "average speed"
UNIT = "km/h"
DELTA_UNIT = "km/h"
MAX_PLAUSIBLE_KMH = 150.0


def _verdict(cleaned: CleanedEntity) -> tuple[bool, str | None]:
    if cleaned.empty:
        return False, "no data in this window"
    if cleaned.coverage["low_coverage"]:
        return False, f"coverage {cleaned.coverage['coverage_pct']}% is below {cleaning.MIN_COVERAGE_PCT:.0f}%"
    if ATTR not in cleaned.readings:
        return False, f"no {ATTR} attribute"
    if cleaned.readings[ATTR].nunique() <= 1:
        return False, "flat line: one value all month, sensor not updating"
    return True, None


def prepare_traffic(start: datetime, end: datetime) -> dict[str, RhythmSeries]:
    """All traffic sensors for [start, end), keyed by street id (e.g. 'AmKroekentor')."""
    try:
        limits = {e["id"]: e.get("speedLimit") for e in orion.list_entities(ENTITY_TYPE)}
    except orion.OrionError as e:
        print(f"[TRAFFIC] WARNING: Orion unavailable ({e}); no speed limits")
        limits = {}
    ents = quantumleap.list_entities()
    entity_ids = sorted(ents.query("entityType == @ENTITY_TYPE")["entityId"])

    out: dict[str, RhythmSeries] = {}
    for entity_id in entity_ids:
        raw = quantumleap.fetch_history(entity_id, start, end)
        n_zero = n_implausible = 0
        if not raw.empty and ATTR in raw:
            zero = raw[ATTR] <= 0
            implausible = raw[ATTR] > MAX_PLAUSIBLE_KMH
            n_zero, n_implausible = int(zero.sum()), int(implausible.sum())
            raw = raw[~zero & ~implausible]
        cleaned = cleaning.clean(entity_id, raw, start, end)
        usable, reason = _verdict(cleaned)
        key = entity_id.removeprefix(ID_PREFIX)
        name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key)      # "WaltherRathenauStrasse" -> "Walther Rathenau Strasse"
        limit = limits.get(entity_id)
        note = (f"{n_implausible} implausible readings above {MAX_PLAUSIBLE_KMH:.0f} km/h dropped"
                if n_implausible else None)
        out[key] = RhythmSeries(
            key=key,
            name=name,
            values=cleaned.hourly[ATTR] if ATTR in cleaned.hourly else pd.Series(dtype=float, index=cleaned.hourly.index),
            coverage=cleaned.coverage,
            usable=usable,
            exclude_reason=reason,
            meta={"speed_limit_kmh": int(limit) if isinstance(limit, (int, float)) else None,
                  "zero_readings_ignored": n_zero, "implausible_dropped": n_implausible,
                  "note": note, "raw_rows": cleaned.raw_rows, "rows": cleaned.rows},
        )
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m data.traffic <YYYY-MM>")
        return 1
    start, end = month_window(*parse_month(sys.argv[1]))
    sensors = prepare_traffic(start, end)
    print(f"\n{'street':28s} {'limit':>5s} {'raw':>6s} {'rows':>5s} {'zero':>5s} {'>150':>5s} {'cover':>6s} {'mean':>6s}  verdict")
    for key, s in sensors.items():
        m = s.meta
        mean = s.values.mean()
        print(f"{key:28s} {str(m['speed_limit_kmh']):>5s} {m['raw_rows']:6d} {m['rows']:5d} {m['zero_readings_ignored']:5d} "
              f"{m['implausible_dropped']:5d} {s.coverage['coverage_pct']:5.1f}% {mean if pd.notna(mean) else 0:6.1f}  "
              f"{'usable' if s.usable else 'EXCLUDED: ' + s.exclude_reason}")
    print(f"\n{sum(s.usable for s in sensors.values())} of {len(sensors)} sensors usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
