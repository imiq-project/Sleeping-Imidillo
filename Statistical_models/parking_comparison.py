
# Parking comparison: the report month against the month before.

from __future__ import annotations

import json
import sys

import pandas as pd
import statsmodels.formula.api as smf

import config
from data.parking import ParkingLot, prepare_parking
from data.timewindow import month_label, month_window, parse_month, previous_of
from Statistical_models.parking_hourly_regression import (
    ALPHA, HAC_MAXLAGS, MIN_HOURS, _frame, _p, analyse_parking,
)


def _stacked(prev: ParkingLot, cur: ParkingLot) -> pd.DataFrame:
    return pd.concat([_frame(prev).assign(is_current=0),
                      _frame(cur).assign(is_current=1)])


def _shift(df: pd.DataFrame) -> dict:
    """Change in occupancy (pp), hour-of-day and weekday held constant."""
    has_weekend = df["is_weekend"].nunique() == 2
    formula = ("occ_pct ~ C(hour) + is_weekend + is_current" if has_weekend
               else "occ_pct ~ C(hour) + is_current")
    fit = smf.ols(formula, data=df).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    lo, hi = fit.conf_int().loc["is_current"]
    p = float(fit.pvalues["is_current"])
    return {
        "pp": round(float(fit.params["is_current"]), 1),
        "ci95": [round(float(lo), 1), round(float(hi), 1)],
        "p_value": _p(p),
        "significant": bool(p < ALPHA),
    }


def _weekend_change(df: pd.DataFrame) -> dict | None:
    """Did the weekend effect change between the months? None if either month
    lacks weekend or weekday hours."""
    per_month = df.groupby("is_current")["is_weekend"].nunique()
    if len(per_month) < 2 or (per_month < 2).any():
        return None
    fit = smf.ols("occ_pct ~ C(hour) + is_weekend + is_current + is_current:is_weekend",
                  data=df).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    term = "is_current:is_weekend"
    lo, hi = fit.conf_int().loc[term]
    p = float(fit.pvalues[term])
    return {
        "change_pp": round(float(fit.params[term]), 1),
        "ci95": [round(float(lo), 1), round(float(hi), 1)],
        "p_value": _p(p),
        "significant": bool(p < ALPHA),
    }


def _compare_lot(prev: ParkingLot, cur: ParkingLot, prev_f: dict, cur_f: dict) -> dict:
    df = _stacked(prev, cur)
    weekend = _weekend_change(df)
    prev_we = prev_f["weekend_effect"]["pp"] if prev_f["weekend_effect"] else None
    cur_we = cur_f["weekend_effect"]["pp"] if cur_f["weekend_effect"] else None
    return {
        "name": cur.name,
        "n_hours": {"previous": prev_f["n_hours"], "current": cur_f["n_hours"]},
        "coverage_pct": {"previous": prev_f["coverage_pct"], "current": cur_f["coverage_pct"]},
        "mean_occ_pct": {
            "previous": prev_f["mean_occ_pct"],
            "current": cur_f["mean_occ_pct"],
            "change_pp": round(cur_f["mean_occ_pct"] - prev_f["mean_occ_pct"], 1),
        },
        "adjusted_shift": _shift(df),
        "weekend_effect_pp": {
            "previous": prev_we,
            "current": cur_we,
            "change": weekend,
        },
        "peak": {
            "previous": {"hour": prev_f["peak_hour"], "occ_pct": prev_f["peak_occ_pct"]},
            "current": {"hour": cur_f["peak_hour"], "occ_pct": cur_f["peak_occ_pct"]},
        },
        "max_occ_pct": {"previous": prev_f["max_occ_pct"], "current": cur_f["max_occ_pct"]},
    }


def compare_parking(prev_lots: dict[str, ParkingLot], cur_lots: dict[str, ParkingLot],
                    prev_ym: tuple[int, int], cur_ym: tuple[int, int]) -> dict:
    """Facts for the 'versus last month' part of the parking section."""
    prev_facts = analyse_parking(prev_lots, *prev_ym)["lots"]
    cur_facts = analyse_parking(cur_lots, *cur_ym)["lots"]

    compared: dict[str, dict] = {}
    not_comparable: dict[str, str] = {}
    for short_id in sorted(set(prev_lots) | set(cur_lots)):
        if short_id not in cur_facts and short_id not in prev_facts:
            not_comparable[short_id] = "not modelled in either month"
        elif short_id not in cur_facts:
            not_comparable[short_id] = "not modelled this month (see parking_patterns.skipped)"
        elif short_id not in prev_facts:
            not_comparable[short_id] = "not modelled last month, nothing to compare against"
        else:
            compared[short_id] = _compare_lot(prev_lots[short_id], cur_lots[short_id],
                                              prev_facts[short_id], cur_facts[short_id])

    fuller = [k for k, v in compared.items() if v["adjusted_shift"]["significant"] and v["adjusted_shift"]["pp"] > 0]
    emptier = [k for k, v in compared.items() if v["adjusted_shift"]["significant"] and v["adjusted_shift"]["pp"] < 0]
    unchanged = [k for k in compared if k not in fuller and k not in emptier]

    return {
        "section": "parking_comparison",
        "month": f"{cur_ym[0]}-{cur_ym[1]:02d}",
        "month_label": month_label(*cur_ym),
        "previous_month": f"{prev_ym[0]}-{prev_ym[1]:02d}",
        "previous_month_label": month_label(*prev_ym),
        "timezone": config.LOCAL_TZ,
        "method": ("Both months stacked; occ_pct ~ C(hour) + is_weekend + is_current with "
                   "Newey-West (HAC, 24 lags) errors. is_current = change in occupancy (pp) at "
                   "equal hour and weekday. Weekend change from the is_current:is_weekend "
                   "interaction. alpha 0.05."),
        "summary": {
            "lots_compared": len(compared),
            "significantly_fuller": fuller,
            "significantly_emptier": emptier,
            "no_significant_change": unchanged,
        },
        "lots": compared,
        "not_comparable": not_comparable,
    }

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m Statistical_models.parking_comparison <YYYY-MM>")
        return 1
    year, month = parse_month(sys.argv[1])
    py, pm = previous_of(year, month)

    print(f"[COMPARISON] fetching {month_label(py, pm)} and {month_label(year, month)} ...")
    prev_lots = prepare_parking(*month_window(py, pm))
    cur_lots = prepare_parking(*month_window(year, month))
    facts = compare_parking(prev_lots, cur_lots, (py, pm), (year, month))

    print(f"\n{'lot':26s} {'mean prev->cur':>16s} {'adj. shift':>14s} {'weekend prev->cur':>20s} {'peak prev->cur':>20s}")
    for k, v in facts["lots"].items():
        m, s, w, p = v["mean_occ_pct"], v["adjusted_shift"], v["weekend_effect_pp"], v["peak"]
        w_txt = (f"{w['previous']:+.1f} -> {w['current']:+.1f}"
                 f"{' *' if w['change'] and w['change']['significant'] else ''}"
                 if w["previous"] is not None and w["current"] is not None else "n/a")
        print(f"{k:26s} {m['previous']:5.1f} -> {m['current']:5.1f} ({m['change_pp']:+.1f}) "
              f"{s['pp']:+6.1f} pp {'*' if s['significant'] else 'ns':>2s} "
              f"{w_txt:>20s} "
              f"{p['previous']['hour']:02d}:00 {p['previous']['occ_pct']:3.0f}% -> "
              f"{p['current']['hour']:02d}:00 {p['current']['occ_pct']:3.0f}%")
    for k, reason in facts["not_comparable"].items():
        print(f"{k:26s} {reason}")
    sm = facts["summary"]
    print(f"\nfuller: {sm['significantly_fuller'] or '-'} | emptier: {sm['significantly_emptier'] or '-'} "
          f"| unchanged: {sm['no_significant_change'] or '-'}")

    out_dir = config.OUTPUT_DIR / facts["month"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "facts_parking_comparison.json"
    out_file.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfacts written to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())