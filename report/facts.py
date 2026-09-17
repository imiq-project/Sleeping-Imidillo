"""
Facts assembly: everything the report may say, in one JSON.

This is the only thing the LLM writer ever sees. It runs the statistical
models for every data family, adds highlights, data quality and a glossary.
Nothing in here is prose; every number is rounded and carries its
uncertainty where one exists.

Structure:
  report        month, covered period (the running month is partial), previous month
  highlights    the most notable facts across families, ranked deterministically
  families      one entry per data family, all with the same shape:
                  label, variable, unit, delta_unit, higher_means, lower_means,
                  patterns   (Statistical_models.rhythm.analyse_rhythm)
                  comparison (Statistical_models.rhythm.compare_rhythm)
                  quality    (usable / excluded / silent / notes)
                  thresholds (air quality only: limit exceedances)
                Families today: parking, traffic, air_no2, air_o3, air_pm10, air_pm25
  glossary      meaning of the recurring field names

Run manually:
    python -m report.facts 2026-08        # writes output/2026-08/facts.json
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from data import air_quality, parking, traffic
from data.timewindow import (
    days_covered, month_label, month_window, parse_month, period_label, previous_of, report_window,
)
from Statistical_models import thresholds
from Statistical_models.rhythm import RhythmSeries, analyse_rhythm, compare_rhythm, quality

GLOSSARY = {
    "unit / delta_unit": "every family states the unit of its values and of its differences (parking: % and percentage points)",
    "higher_means / lower_means": "what a higher or lower value means for that family, e.g. fuller / emptier, faster / slower, more polluted / cleaner",
    "ci95": "95% confidence interval [low, high]; if it contains 0 the effect is not distinguishable from zero",
    "p_value": "probability of seeing an effect this large if there were none; below 0.05 counts as significant",
    "significant": "true when p_value < 0.05",
    "weekend_effect.delta": "value on Saturday/Sunday minus value on a weekday at the same hour",
    "adjusted_shift.delta": "this month minus previous month at equal hour and weekday; the honest month-on-month change",
    "peak_hour / trough_hour": "hour of a typical weekday with the highest / lowest fitted value, local time, 0-23",
    "profile_weekday / profile_weekend": "fitted value for hours 0..23 of a typical weekday / weekend day; null = hour never observed",
    "coverage_pct": "share of hours in the period with at least one reading; below the threshold the sensor is excluded",
    "longest_gap_h": "longest run of consecutive hours without any reading",
    "r_squared": "share of hourly variation explained by hour-of-day and weekend; low means other things drive the series",
    "pooled.weekend_effect_differs_between_series": "true when the weekend effect is not the same for all series of the family",
    "thresholds.count_over": "how many hours or days were above a health limit; allowed_per_year is the EU tolerance where one exists",
    "report.partial": "true when the report was written before the month ended; it then covers report.period_label only, while the comparison month is complete",
}


# --------------------------------------------------------------------------- #
# One family                                                                   #
# --------------------------------------------------------------------------- #

def _family(label: str, higher: str, lower: str, variable: str, unit: str, delta_unit: str,
            prev: dict[str, RhythmSeries], cur: dict[str, RhythmSeries],
            prev_ym: tuple[int, int], cur_ym: tuple[int, int]) -> dict:
    kw = {"variable": variable, "unit": unit, "delta_unit": delta_unit}
    return {
        "label": label,
        "variable": variable,
        "unit": unit,
        "delta_unit": delta_unit,
        "higher_means": higher,
        "lower_means": lower,
        "patterns": analyse_rhythm(cur, year=cur_ym[0], month=cur_ym[1], **kw),
        "comparison": compare_rhythm(prev, cur, prev_ym, cur_ym, **kw),
        "quality": quality(cur),
    }


def _air_thresholds(stations: dict[str, RhythmSeries], attr: str) -> dict:
    return {k: thresholds.exceedances(s.values, attr) for k, s in stations.items() if s.usable}


# --------------------------------------------------------------------------- #
# Highlights: ranked deterministically, no LLM                                 #
# --------------------------------------------------------------------------- #

def _highlights(families: dict, limit: int = 6) -> list[dict]:
    items = []
    for fkey, fam in families.items():
        for skey, c in fam["comparison"]["series"].items():
            a = c["adjusted_shift"]
            if a["significant"]:
                half = max((a["ci95"][1] - a["ci95"][0]) / 2, 1e-9)
                items.append({
                    "kind": "month_on_month_change", "family": fkey, "family_label": fam["label"],
                    "series": skey, "name": c["name"], "delta": a["delta"], "delta_unit": fam["delta_unit"],
                    "direction": fam["higher_means"] if a["delta"] > 0 else fam["lower_means"],
                    "previous_mean": c["mean"]["previous"], "current_mean": c["mean"]["current"],
                    "strength": round(abs(a["delta"]) / half, 1),
                })
            w = c["weekend_effect"]["change"]
            if w and w["significant"]:
                half = max((w["ci95"][1] - w["ci95"][0]) / 2, 1e-9)
                items.append({
                    "kind": "weekend_effect_change", "family": fkey, "family_label": fam["label"],
                    "series": skey, "name": c["name"], "delta": w["delta"], "delta_unit": fam["delta_unit"],
                    "weekend_effect_previous": c["weekend_effect"]["previous"],
                    "weekend_effect_current": c["weekend_effect"]["current"],
                    "strength": round(abs(w["delta"]) / half, 1),
                })
        for skey, limits in fam.get("thresholds", {}).items():
            for t in limits:
                if t["count_over"] > 0:
                    items.append({
                        "kind": "threshold_exceeded", "family": fkey, "family_label": fam["label"],
                        "series": skey, "name": skey, "limit": t["limit"], "limit_value": t["value"],
                        "unit": fam["unit"], "count_over": t["count_over"], "count_unit": t["count_unit"],
                        "worst": t["worst"], "strength": 10.0 + t["count_over"],
                    })
    items.sort(key=lambda x: -x["strength"])
    picked, per_kind = [], {}
    for item in items:                       # at most 3 of a kind, so one theme cannot crowd out the rest
        if per_kind.get(item["kind"], 0) < 3:
            picked.append(item)
            per_kind[item["kind"]] = per_kind.get(item["kind"], 0) + 1
        if len(picked) == limit:
            break
    return picked


# --------------------------------------------------------------------------- #
# The one function the monthly job calls                                       #
# --------------------------------------------------------------------------- #

def build_facts(year: int, month: int, now: datetime | None = None) -> dict:
    """Run every model for the report month and bundle the results.

    The report month is covered up to `now` (the job runs on the last Monday,
    before the month ends); the comparison month is always complete."""
    py, pm = previous_of(year, month)
    cur_ym, prev_ym = (year, month), (py, pm)
    start, end = report_window(year, month, now)
    full_start, full_end = month_window(year, month)
    prev_start, prev_end = month_window(py, pm)
    partial = end < full_end
    print(f"[FACTS] {period_label(start, end)}{' (month still running)' if partial else ''} "
          f"vs {month_label(py, pm)} (full)")

    print("[FACTS] parking ...")
    cur_park, prev_park = parking.prepare_parking(start, end), parking.prepare_parking(prev_start, prev_end)
    print("[FACTS] traffic ...")
    cur_tr, prev_tr = traffic.prepare_traffic(start, end), traffic.prepare_traffic(prev_start, prev_end)
    print("[FACTS] air quality ...")
    cur_air, prev_air = air_quality.prepare_air_quality(start, end), air_quality.prepare_air_quality(prev_start, prev_end)

    print("[FACTS] models ...")
    families = {
        "parking": _family("Parking occupancy", "fuller", "emptier", parking.VARIABLE, parking.UNIT,
                           parking.DELTA_UNIT, prev_park, cur_park, prev_ym, cur_ym),
        "traffic": _family("Traffic average speed", "faster (less congested)", "slower (more congested)",
                           traffic.VARIABLE, traffic.UNIT, traffic.DELTA_UNIT, prev_tr, cur_tr, prev_ym, cur_ym),
    }
    for attr, (label, unit) in air_quality.POLLUTANTS.items():
        fam = _family(f"Air quality: {label}", "more polluted", "cleaner", f"{label} concentration",
                      unit, air_quality.DELTA_UNIT, prev_air[attr], cur_air[attr], prev_ym, cur_ym)
        fam["thresholds"] = _air_thresholds(cur_air[attr], attr)
        families[f"air_{attr}"] = fam

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
            "previous_month_days": (prev_end - prev_start).days,
            "timezone": config.LOCAL_TZ,
            "generated_at": datetime.now(ZoneInfo(config.LOCAL_TZ)).strftime("%Y-%m-%d %H:%M %Z"),
            "families": list(families),
        },
        "highlights": _highlights(families),
        "families": families,
        "glossary": GLOSSARY,
    }


def slim_facts(facts: dict) -> dict:
    """The copy the LLMs get: without the 24-hour profiles (they are for the
    charts) so the prompt stays small."""
    slim = copy.deepcopy(facts)
    for fam in slim["families"].values():
        for s in fam["patterns"]["series"].values():
            s.pop("profile_weekday", None)
            s.pop("profile_weekend", None)
    return slim


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
    facts = build_facts(*parse_month(sys.argv[1]))
    path = save_facts(facts)
    r = facts["report"]
    print(f"\n[FACTS] {r['period_label']} ({r['days_covered']}/{r['days_in_month']} days) vs {r['previous_month_label']}")
    for key, fam in facts["families"].items():
        q, c = fam["quality"], fam["comparison"]["summary"]
        print(f"[FACTS] {key:10s} {len(q['usable'])}/{q['total']} usable, {len(fam['patterns']['series'])} modelled, "
              f"{c['series_compared']} compared: higher {c['significantly_higher'] or '-'}, lower {c['significantly_lower'] or '-'}")
    print(f"[FACTS] highlights: {[(h['kind'], h['family'], h['series']) for h in facts['highlights']]}")
    print(f"[FACTS] written {path} ({path.stat().st_size / 1024:.1f} KB; slim for LLM "
          f"{len(json.dumps(slim_facts(facts))) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
