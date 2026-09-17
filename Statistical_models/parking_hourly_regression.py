# Parking model: hour-of-day and weekend effects per lot.

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson

import config
from data.parking import ParkingLot, prepare_parking
from data.timewindow import month_label, month_window, parse_month

ALPHA = 0.05          # significance threshold
HAC_MAXLAGS = 24      # one day of hourly lags for the Newey-West errors
MIN_HOURS = 24 * 7    # less than a week of hourly data = do not model


def _frame(lot: ParkingLot) -> pd.DataFrame:
    """Hourly occupancy with the two regressors the model needs."""
    df = lot.occ_pct.dropna().to_frame("occ_pct")
    df["hour"] = df.index.hour
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def _fit(df: pd.DataFrame):
    has_weekend = df["is_weekend"].nunique() == 2
    formula = "occ_pct ~ C(hour) + is_weekend" if has_weekend else "occ_pct ~ C(hour)"
    model = smf.ols(formula, data=df)
    dw = float(durbin_watson(model.fit().resid))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    return fit, dw, has_weekend


def _profiles(fit, has_weekend: bool) -> tuple[list[float | None], list[float | None]]:
    """Fitted occupancy for each of the 24 hours: weekday and weekend."""
    base = float(fit.params["Intercept"])
    weekend_shift = float(fit.params["is_weekend"]) if has_weekend else 0.0
    weekday, weekend = [], []
    for h in range(24):
        coef = 0.0 if h == 0 else fit.params.get(f"C(hour)[T.{h}]", np.nan)
        if pd.isna(coef):                      # hour never observed this month
            weekday.append(None)
            weekend.append(None)
        else:
            weekday.append(round(base + float(coef), 1))
            weekend.append(round(base + float(coef) + weekend_shift, 1))
    return weekday, weekend


def _p(x: float) -> float:
    """p-values rounded to 3 significant digits, JSON-safe."""
    return float(f"{x:.3g}")


def _lot_facts(lot: ParkingLot, df: pd.DataFrame) -> dict:
    fit, dw, has_weekend = _fit(df)
    weekday, weekend = _profiles(fit, has_weekend)

    known = [(h, v) for h, v in enumerate(weekday) if v is not None]
    peak_hour, peak_val = max(known, key=lambda hv: hv[1])
    trough_hour, trough_val = min(known, key=lambda hv: hv[1])

    hour_terms = [t for t in fit.params.index if t.startswith("C(hour)")]
    n_sig_hours = int(sum(fit.pvalues[t] < ALPHA for t in hour_terms))

    facts = {
        "name": lot.name,
        "capacity": lot.capacity,
        "capacity_note": lot.capacity_note,
        "n_hours": int(len(df)),
        "coverage_pct": lot.cleaned.coverage["coverage_pct"],
        "longest_gap_h": lot.cleaned.coverage["longest_gap_h"],
        "mean_occ_pct": round(float(df["occ_pct"].mean()), 1),
        "max_occ_pct": round(float(df["occ_pct"].max()), 1),
        "peak_hour": peak_hour,
        "peak_occ_pct": peak_val,
        "trough_hour": trough_hour,
        "trough_occ_pct": trough_val,
        "hours_significantly_different_from_midnight": n_sig_hours,
        "profile_weekday_pct": weekday,
        "profile_weekend_pct": weekend,
        "r_squared": round(float(fit.rsquared), 3),
        "durbin_watson": round(dw, 2),
    }
    if has_weekend:
        ci_low, ci_high = fit.conf_int().loc["is_weekend"]
        p = float(fit.pvalues["is_weekend"])
        facts["weekend_effect"] = {
            "pp": round(float(fit.params["is_weekend"]), 1),
            "ci95": [round(float(ci_low), 1), round(float(ci_high), 1)],
            "p_value": _p(p),
            "significant": bool(p < ALPHA),
        }
    else:
        facts["weekend_effect"] = None   # no weekend hours observed this month
    return facts


def _pooled_facts(frames: dict[str, pd.DataFrame]) -> dict | None:
    """Does the weekend effect differ between lots? Joint Wald test on the
    lot x weekend interaction, hour-of-day controlled."""
    usable = {k: v for k, v in frames.items() if v["is_weekend"].nunique() == 2}
    if len(usable) < 2:
        return None
    pooled = pd.concat([v.assign(lot=k) for k, v in usable.items()], ignore_index=True)
    fit = smf.ols("occ_pct ~ C(lot) + C(hour) + is_weekend + C(lot):is_weekend",
                  data=pooled).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    terms = [t for t in fit.params.index if "is_weekend" in t and "C(lot)" in t]
    wald = fit.wald_test(" = ".join(terms) + " = 0", scalar=True)
    p = float(np.asarray(wald.pvalue).squeeze())
    return {
        "lots": sorted(usable),
        "n_hours": int(len(pooled)),
        "f_stat": round(float(np.asarray(wald.statistic).squeeze()), 1),
        "p_value": _p(p),
        "weekend_effect_differs_by_lot": bool(p < ALPHA),
    }


def analyse_parking(lots: dict[str, ParkingLot], year: int, month: int) -> dict:
    """Facts for the parking section of the report month."""
    modelled: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}

    for short_id, lot in lots.items():
        if not lot.usable:
            skipped[short_id] = lot.exclude_reason
            continue
        df = _frame(lot)
        if len(df) < MIN_HOURS:
            skipped[short_id] = f"only {len(df)} hourly values, need {MIN_HOURS}"
            continue
        frames[short_id] = df
        modelled[short_id] = _lot_facts(lot, df)

    return {
        "section": "parking_patterns",
        "month": f"{year}-{month:02d}",
        "month_label": month_label(year, month),
        "timezone": config.LOCAL_TZ,
        "method": ("Per lot: OLS of hourly occupancy% on hour-of-day dummies and a weekend "
                   "dummy, Newey-West (HAC, 24 lags) standard errors, alpha 0.05. Pooled: "
                   "lot x weekend interaction, joint Wald test."),
        "lots": modelled,
        "skipped": skipped,
        "pooled": _pooled_facts(frames),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m Statistical_models.parking_hourly_regression <YYYY-MM>")
        return 1
    year, month = parse_month(sys.argv[1])
    start, end = month_window(year, month)

    print(f"[PARKING MODEL] fetching {month_label(year, month)} ...")
    lots = prepare_parking(start, end)
    facts = analyse_parking(lots, year, month)

    print(f"\n{'lot':16s} {'hours':>5s} {'mean%':>6s} {'peak':>10s} {'trough':>10s} {'weekend':>14s}  R2")
    for short_id, f in facts["lots"].items():
        we = f["weekend_effect"]
        we_txt = (f"{we['pp']:+.1f} pp {'*' if we['significant'] else 'ns'}") if we else "n/a"
        print(f"{short_id:16s} {f['n_hours']:5d} {f['mean_occ_pct']:6.1f} "
              f"{f['peak_hour']:02d}:00 {f['peak_occ_pct']:4.0f}% "
              f"{f['trough_hour']:02d}:00 {f['trough_occ_pct']:4.0f}% {we_txt:>14s}  {f['r_squared']:.2f}")
    for short_id, reason in facts["skipped"].items():
        print(f"{short_id:16s} skipped: {reason}")
    if facts["pooled"]:
        p = facts["pooled"]
        print(f"\npooled: weekend effect differs by lot = {p['weekend_effect_differs_by_lot']} "
              f"(F={p['f_stat']}, p={p['p_value']})")

    out_dir = config.OUTPUT_DIR / facts["month"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "facts_parking_patterns.json"
    out_file.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfacts written to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())