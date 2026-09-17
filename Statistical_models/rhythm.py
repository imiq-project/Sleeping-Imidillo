"""
Rhythm model: hour-of-day and weekend effects for ANY hourly series.

One model, many data families. Parking occupancy, traffic speed and air
pollutants all come in as a dict of RhythmSeries (one per lot / street /
station) and go out as the same facts structure, so the writer, verifier
and charts treat them alike.

THE MODEL, IN PLAIN TERMS
-------------------------
For each series we fit   value = base + effect(hour) + effect(weekend) + noise
That is 25 numbers: the weekday-midnight baseline, how much each other hour
differs from it, and how much a weekend day differs at any hour. Every number
carries a confidence interval and a p-value, so the report can say "the
weekend effect is real" or "this difference is noise". Hourly series are
strongly autocorrelated (one hour looks like the next), so standard errors
are Newey-West (HAC); plain OLS errors would overstate certainty.

From the fitted model we read the busiest/highest and quietest/lowest hour
off the 24-hour fitted profile (midnight included) and count the hours that
differ from midnight significantly. One pooled model across all series tests
whether the weekend effect differs between them (series x weekend
interaction, joint Wald test; a chi-square under HAC).

COMPARISON WITH THE PREVIOUS MONTH
----------------------------------
Both months stacked with a flag is_current and the model
    value ~ C(hour) + is_weekend + is_current
The is_current coefficient is the change at equal hour and weekday: a month
with more Sundays, or a sensor silent at night, cannot fake a change. A
second fit with is_current:is_weekend says whether the weekend effect moved.

INPUT   dict[key, RhythmSeries]; values = hourly series on the report window,
        local time, NaN where no reading. Preparation modules (data/parking.py,
        data/traffic.py, data/air_quality.py) build these.
OUTPUT  facts dicts, JSON-serialisable, every number rounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson

import config
from data.timewindow import month_label

ALPHA = 0.05          # significance threshold
HAC_MAXLAGS = 24      # one day of hourly lags for the Newey-West errors
MIN_HOURS = 24 * 7    # less than a week of hourly values = do not model


@dataclass
class RhythmSeries:
    key: str                      # short id, e.g. "Ulrichshaus", "AmKroekentor", "West"
    name: str                     # display name for the report
    values: pd.Series             # hourly, local tz, NaN where no reading
    coverage: dict                # from data/cleaning.py
    usable: bool
    exclude_reason: str | None = None
    meta: dict = field(default_factory=dict)   # capacity, speed limit, ... (copied into facts)


# --------------------------------------------------------------------------- #
# Fitting one series                                                           #
# --------------------------------------------------------------------------- #

def _frame(s: RhythmSeries) -> pd.DataFrame:
    df = s.values.dropna().to_frame("y")
    df["hour"] = df.index.hour
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    return df


def _fit(df: pd.DataFrame):
    has_weekend = df["is_weekend"].nunique() == 2
    formula = "y ~ C(hour) + is_weekend" if has_weekend else "y ~ C(hour)"
    model = smf.ols(formula, data=df)
    dw = float(durbin_watson(model.fit().resid))
    fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    return fit, dw, has_weekend


def _p(x: float) -> float:
    return float(f"{x:.3g}")


def _effect(fit, term: str, decimals: int) -> dict:
    lo, hi = fit.conf_int().loc[term]
    p = float(fit.pvalues[term])
    return {
        "delta": round(float(fit.params[term]), decimals),
        "ci95": [round(float(lo), decimals), round(float(hi), decimals)],
        "p_value": _p(p),
        "significant": bool(p < ALPHA),
    }


def _profiles(fit, has_weekend: bool, decimals: int) -> tuple[list, list]:
    base = float(fit.params["Intercept"])
    shift = float(fit.params["is_weekend"]) if has_weekend else 0.0
    weekday, weekend = [], []
    for h in range(24):
        coef = 0.0 if h == 0 else fit.params.get(f"C(hour)[T.{h}]", np.nan)
        if pd.isna(coef):
            weekday.append(None)
            weekend.append(None)
        else:
            weekday.append(round(base + float(coef), decimals))
            weekend.append(round(base + float(coef) + shift, decimals))
    return weekday, weekend


def _series_facts(s: RhythmSeries, df: pd.DataFrame, decimals: int) -> dict:
    fit, dw, has_weekend = _fit(df)
    weekday, weekend = _profiles(fit, has_weekend, decimals)
    known = [(h, v) for h, v in enumerate(weekday) if v is not None]
    peak_hour, peak_val = max(known, key=lambda hv: hv[1])
    trough_hour, trough_val = min(known, key=lambda hv: hv[1])
    hour_terms = [t for t in fit.params.index if t.startswith("C(hour)")]
    facts = {
        "name": s.name,
        **s.meta,
        "n_hours": int(len(df)),
        "coverage_pct": s.coverage["coverage_pct"],
        "longest_gap_h": s.coverage["longest_gap_h"],
        "mean": round(float(df["y"].mean()), decimals),
        "min": round(float(df["y"].min()), decimals),
        "max": round(float(df["y"].max()), decimals),
        "peak_hour": peak_hour,
        "peak_value": peak_val,
        "trough_hour": trough_hour,
        "trough_value": trough_val,
        "hours_significantly_different_from_midnight": int(sum(fit.pvalues[t] < ALPHA for t in hour_terms)),
        "profile_weekday": weekday,
        "profile_weekend": weekend,
        "r_squared": round(float(fit.rsquared), 3),
        "durbin_watson": round(dw, 2),
        "weekend_effect": _effect(fit, "is_weekend", decimals) if has_weekend else None,
    }
    return facts


def _pooled(frames: dict[str, pd.DataFrame]) -> dict | None:
    usable = {k: v for k, v in frames.items() if v["is_weekend"].nunique() == 2}
    if len(usable) < 2:
        return None
    pooled = pd.concat([v.assign(series=k) for k, v in usable.items()], ignore_index=True)
    fit = smf.ols("y ~ C(series) + C(hour) + is_weekend + C(series):is_weekend",
                  data=pooled).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    terms = [t for t in fit.params.index if "is_weekend" in t and "C(series)" in t]
    wald = fit.wald_test(" = ".join(terms) + " = 0", scalar=True)
    p = float(np.asarray(wald.pvalue).squeeze())
    return {
        "series": sorted(usable),
        "n_hours": int(len(pooled)),
        "wald_chi2": round(float(np.asarray(wald.statistic).squeeze()), 1),
        "p_value": _p(p),
        "weekend_effect_differs_between_series": bool(p < ALPHA),
    }


# --------------------------------------------------------------------------- #
# Public: patterns for one month                                               #
# --------------------------------------------------------------------------- #

def analyse_rhythm(series: dict[str, RhythmSeries], *, year: int, month: int,
                   variable: str, unit: str, delta_unit: str, decimals: int = 1) -> dict:
    modelled, frames, skipped = {}, {}, {}
    for key, s in series.items():
        if not s.usable:
            skipped[key] = s.exclude_reason
            continue
        df = _frame(s)
        if len(df) < MIN_HOURS:
            skipped[key] = f"only {len(df)} hourly values, need {MIN_HOURS}"
            continue
        frames[key] = df
        modelled[key] = _series_facts(s, df, decimals)
    return {
        "variable": variable,
        "unit": unit,
        "delta_unit": delta_unit,
        "month": f"{year}-{month:02d}",
        "month_label": month_label(year, month),
        "timezone": config.LOCAL_TZ,
        "method": (f"Per series: OLS of hourly {variable} on hour-of-day dummies and a weekend dummy, "
                   "Newey-West (HAC, 24 lags) standard errors, alpha 0.05. Pooled: series x weekend "
                   "interaction, joint Wald chi-square test."),
        "series": modelled,
        "skipped": skipped,
        "pooled": _pooled(frames),
    }


# --------------------------------------------------------------------------- #
# Public: this month versus the previous one                                   #
# --------------------------------------------------------------------------- #

def _shift(df: pd.DataFrame, decimals: int) -> dict:
    has_weekend = df["is_weekend"].nunique() == 2
    formula = "y ~ C(hour) + is_weekend + is_current" if has_weekend else "y ~ C(hour) + is_current"
    fit = smf.ols(formula, data=df).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    return _effect(fit, "is_current", decimals)


def _weekend_change(df: pd.DataFrame, decimals: int) -> dict | None:
    per_month = df.groupby("is_current")["is_weekend"].nunique()
    if len(per_month) < 2 or (per_month < 2).any():
        return None
    fit = smf.ols("y ~ C(hour) + is_weekend + is_current + is_current:is_weekend",
                  data=df).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS})
    return _effect(fit, "is_current:is_weekend", decimals)


def compare_rhythm(prev: dict[str, RhythmSeries], cur: dict[str, RhythmSeries],
                   prev_ym: tuple[int, int], cur_ym: tuple[int, int], *,
                   variable: str, unit: str, delta_unit: str, decimals: int = 1) -> dict:
    prev_f = analyse_rhythm(prev, year=prev_ym[0], month=prev_ym[1], variable=variable,
                            unit=unit, delta_unit=delta_unit, decimals=decimals)["series"]
    cur_f = analyse_rhythm(cur, year=cur_ym[0], month=cur_ym[1], variable=variable,
                           unit=unit, delta_unit=delta_unit, decimals=decimals)["series"]
    compared, not_comparable = {}, {}
    for key in sorted(set(prev) | set(cur)):
        if key not in cur_f and key not in prev_f:
            not_comparable[key] = "not modelled in either month"
        elif key not in cur_f:
            not_comparable[key] = "not modelled this month (see patterns.skipped)"
        elif key not in prev_f:
            not_comparable[key] = "not modelled last month, nothing to compare against"
        else:
            df = pd.concat([_frame(prev[key]).assign(is_current=0), _frame(cur[key]).assign(is_current=1)])
            p, c = prev_f[key], cur_f[key]
            compared[key] = {
                "name": c["name"],
                "n_hours": {"previous": p["n_hours"], "current": c["n_hours"]},
                "coverage_pct": {"previous": p["coverage_pct"], "current": c["coverage_pct"]},
                "mean": {"previous": p["mean"], "current": c["mean"],
                         "change": round(c["mean"] - p["mean"], decimals)},
                "adjusted_shift": _shift(df, decimals),
                "weekend_effect": {
                    "previous": p["weekend_effect"]["delta"] if p["weekend_effect"] else None,
                    "current": c["weekend_effect"]["delta"] if c["weekend_effect"] else None,
                    "change": _weekend_change(df, decimals),
                },
                "peak": {"previous": {"hour": p["peak_hour"], "value": p["peak_value"]},
                         "current": {"hour": c["peak_hour"], "value": c["peak_value"]}},
                "max": {"previous": p["max"], "current": c["max"]},
            }
    higher = [k for k, v in compared.items() if v["adjusted_shift"]["significant"] and v["adjusted_shift"]["delta"] > 0]
    lower = [k for k, v in compared.items() if v["adjusted_shift"]["significant"] and v["adjusted_shift"]["delta"] < 0]
    return {
        "variable": variable,
        "unit": unit,
        "delta_unit": delta_unit,
        "month": f"{cur_ym[0]}-{cur_ym[1]:02d}",
        "month_label": month_label(*cur_ym),
        "previous_month": f"{prev_ym[0]}-{prev_ym[1]:02d}",
        "previous_month_label": month_label(*prev_ym),
        "method": (f"Both months stacked; {variable} ~ C(hour) + is_weekend + is_current with Newey-West "
                   "(HAC, 24 lags) errors. is_current = change at equal hour and weekday. Weekend change "
                   "from the is_current:is_weekend interaction. alpha 0.05."),
        "summary": {
            "series_compared": len(compared),
            "significantly_higher": higher,
            "significantly_lower": lower,
            "no_significant_change": [k for k in compared if k not in higher and k not in lower],
        },
        "series": compared,
        "not_comparable": not_comparable,
    }


# --------------------------------------------------------------------------- #
# Public: data-quality summary of a family                                     #
# --------------------------------------------------------------------------- #

def quality(series: dict[str, RhythmSeries]) -> dict:
    usable = {k: v for k, v in series.items() if v.usable}
    return {
        "total": len(series),
        "usable": sorted(usable),
        "excluded": {k: v.exclude_reason for k, v in series.items() if not v.usable},
        "coverage_pct": {k: v.coverage["coverage_pct"] for k, v in series.items()},
        "longest_gap_h": {k: v.coverage["longest_gap_h"] for k, v in usable.items()},
        "silent_all_month": sorted(k for k, v in series.items() if v.coverage["hours_with_data"] == 0),
        "notes": {k: v.meta["note"] for k, v in series.items() if v.meta.get("note")},
    }
