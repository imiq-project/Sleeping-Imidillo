# Facts assembly: everything the report may say, in one JSON.

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from data.parking import ParkingLot, prepare_parking
from data.timewindow import (
    days_covered, month_label, month_window, parse_month, period_label, previous_of, report_window,
)
from Statistical_models.parking_comparison import compare_parking
from Statistical_models.parking_hourly_regression import analyse_parking

GLOSSARY = {
    "pp": "percentage points: an absolute difference between two percentages",
    "occ_pct": "occupancy in percent of the lot's capacity",
    "ci95": "95% confidence interval [low, high]; if it contains 0 the effect is not distinguishable from zero",
    "p_value": "probability of seeing an effect this large if there were none; below 0.05 counts as significant",
    "significant": "true when p_value < 0.05",
    "weekend_effect.pp": "occupancy on Saturday/Sunday minus weekday occupancy at the same hour",
    "adjusted_shift.pp": "this month minus previous month at equal hour and weekday; the honest month-on-month change",
    "peak_hour / trough_hour": "busiest / quietest hour of a typical weekday, local time, 0-23",
    "profile_weekday_pct": "fitted occupancy for hours 0..23 of a typical weekday; null = hour never observed",
    "coverage_pct": "share of hours in the month with at least one reading; below the threshold the sensor is excluded",
    "longest_gap_h": "longest run of consecutive hours without any reading",
    "r_squared": "share of hourly variation explained by hour-of-day and weekend; low means the lot is driven by other things",
    "report.partial": "true when the report was written before the month ended; it then covers report.period_label only, while the comparison month is complete",
}


# --------------------------------------------------------------------------- #
# Data quality section                                                         #
# --------------------------------------------------------------------------- #

def parking_quality(lots: dict[str, ParkingLot]) -> dict:
    usable = {k: v for k, v in lots.items() if v.usable}
    excluded = {k: v.exclude_reason for k, v in lots.items() if not v.usable}
    return {
        "lots_total": len(lots),
        "lots_usable": len(usable),
        "usable": sorted(usable),
        "excluded": excluded,
        "coverage_pct": {k: v.cleaned.coverage["coverage_pct"] for k, v in lots.items()},
        "longest_gap_h": {k: v.cleaned.coverage["longest_gap_h"] for k, v in usable.items()},
        "capacity_notes": {k: v.capacity_note for k, v in lots.items()
                           if "totalSpots from Orion" not in v.capacity_note},
        "silent_all_month": sorted(k for k, v in lots.items() if v.cleaned.empty),
    }


# --------------------------------------------------------------------------- #
# The one function the monthly job calls                                       #
# --------------------------------------------------------------------------- #

def build_facts(year: int, month: int, now: datetime | None = None) -> dict:
    """Run every statistical module for the report month and bundle the results.

    The report month is covered up to `now` (default: the current time) because
    the job runs on the last Monday, before the month ends; the comparison
    month is always complete."""
    py, pm = previous_of(year, month)
    start, end = report_window(year, month, now)          # running month: capped at now
    full_start, full_end = month_window(year, month)
    partial = end < full_end
    print(f"[FACTS] fetching {month_label(py, pm)} (full) and {period_label(start, end)}"
          f"{' (month still running)' if partial else ''} ...")
    cur_lots = prepare_parking(start, end)
    prev_lots = prepare_parking(*month_window(py, pm))

    print("[FACTS] parking patterns ...")
    patterns = analyse_parking(cur_lots, year, month)
    print("[FACTS] parking comparison ...")
    comparison = compare_parking(prev_lots, cur_lots, (py, pm), (year, month))

    return {
        "report": {
            "month": f"{year}-{month:02d}",
            "month_label": month_label(year, month),
            "partial": partial,
            "period_label": period_label(start, end),
            "days_covered": days_covered(start, end),
            "days_in_month": (full_end - full_start).days,
            "previous_month": f"{py}-{pm:02d}",
            "previous_month_label": month_label(py, pm),
            "previous_month_days": (month_window(py, pm)[1] - month_window(py, pm)[0]).days,
            "timezone": config.LOCAL_TZ,
            "generated_at": datetime.now(ZoneInfo(config.LOCAL_TZ)).strftime("%Y-%m-%d %H:%M %Z"),
            "sections": ["data_quality", "parking_patterns", "parking_comparison"],
        },
        "data_quality": {
            "parking": parking_quality(cur_lots),
        },
        "parking_patterns": patterns,
        "parking_comparison": comparison,
        "glossary": GLOSSARY,
    }


def save_facts(facts: dict, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (config.OUTPUT_DIR / facts["report"]["month"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "facts.json"
    path.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_facts(month: str) -> dict:
    """Read a previously saved facts.json, e.g. to re-run only the LLM steps."""
    path = config.OUTPUT_DIR / month / "facts.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Manual check                                                                 #
# --------------------------------------------------------------------------- #

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m report.facts <YYYY-MM>")
        return 1
    year, month = parse_month(sys.argv[1])
    facts = build_facts(year, month)
    path = save_facts(facts)

    q = facts["data_quality"]["parking"]
    p = facts["parking_patterns"]
    c = facts["parking_comparison"]
    print(f"\n[FACTS] {facts['report']['month_label']} vs {facts['report']['previous_month_label']}")
    print(f"[FACTS] parking: {q['lots_usable']}/{q['lots_total']} lots usable, "
          f"{len(q['excluded'])} excluded, {len(q['silent_all_month'])} silent all month")
    print(f"[FACTS] patterns: {len(p['lots'])} lots modelled; pooled weekend test "
          f"{'differs' if p['pooled'] and p['pooled']['weekend_effect_differs_by_lot'] else 'n/a or same'}")
    print(f"[FACTS] comparison: fuller {c['summary']['significantly_fuller'] or '-'}, "
          f"emptier {c['summary']['significantly_emptier'] or '-'}")
    print(f"[FACTS] written {path} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())