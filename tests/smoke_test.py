"""
Smoke test: run every pipeline stage, end to end, in one go.

Each step prints PASS or FAIL with a one-line reason. Network steps talk to
the real Orion / QuantumLeap; they need internet but no API key. The test
fetches one report month (default: the previous calendar month) and reuses
the fetched data across steps.

    python -m tests.smoke_test                  # previous calendar month
    python -m tests.smoke_test 2026-08          # a specific month
    python -m tests.smoke_test 2026-08 --llm    # also run the LLM steps (tokens, ~4 min)

Exit code 0 = everything passed, 1 = at least one step failed.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import date, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_results: list[tuple[str, bool, str]] = []
_cache: dict = {}


def step(name: str):
    """Decorator: run the function, record PASS/FAIL, never abort the suite."""
    def wrap(fn):
        def run(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                ok, note = True, out if isinstance(out, str) else ""
            except AssertionError as e:
                ok, note, out = False, f"assertion: {e}", None
            except Exception as e:  # noqa: BLE001
                ok, note, out = False, f"{type(e).__name__}: {e}", None
                traceback.print_exc()
            dt = time.perf_counter() - t0
            print(f"[{'PASS' if ok else 'FAIL'}] {name:38s} {dt:5.1f}s  {note}")
            _results.append((name, ok, note))
            return out
        return run
    return wrap


# --------------------------------------------------------------------------- #
# 1-3: deploy files, config, time windows                                      #
# --------------------------------------------------------------------------- #

@step("1 deploy/ unit files")
def check_deploy():
    d = ROOT / "deploy"
    for name in ("imidillo-report.service", "imidillo-report.timer", "imidillo-report-test.timer"):
        f = d / name
        assert f.exists(), f"missing {name}"
        assert b"\r" not in f.read_bytes(), f"{name} has Windows line endings (CR); systemd would choke"
    service = (d / "imidillo-report.service").read_text()
    assert "ExecStart=" in service and "jobs.monthly_report" in service, "service ExecStart wrong"
    assert "SuccessExitStatus=2" in service, "service lacks SuccessExitStatus=2"
    timer = (d / "imidillo-report.timer").read_text()
    assert "OnCalendar=Mon *-*~07/1" in timer, "timer OnCalendar not 'last Monday'"
    assert "Persistent=true" in timer, "timer lacks Persistent=true"
    assert "Unit=imidillo-report.service" in (d / "imidillo-report-test.timer").read_text()
    return "3 files, LF endings, last-Monday schedule"


@step("2 config.py")
def check_config():
    import config
    assert (ROOT / ".env").exists(), ".env missing"
    assert ".env" in (ROOT / ".gitignore").read_text(), ".gitignore must list .env"
    assert config.LOCAL_TZ == "Europe/Berlin", config.LOCAL_TZ
    assert config.QL_BASE_URL.startswith("https://") and config.ORION_BASE_URL.startswith("https://")
    assert config.quantumleap_configured(), "quantumleap_configured() is False"
    assert "FIWARE_API_KEY" not in config.missing(), "FIWARE_API_KEY must not be required"
    later = config.missing()
    return (f"QL ready; SMTP ready={config.smtp_configured()}; LLM ready={config.llm_configured()}"
            + (f"; still missing for the full job: {', '.join(later)}" if later else ""))


@step("3 data/timewindow.py")
def check_timewindow():
    from datetime import datetime
    from data.timewindow import (days_covered, month_label, month_window, parse_month,
                                 period_label, previous_month, previous_of, report_window)
    s, e = month_window(2026, 8)
    assert s.isoformat() == "2026-07-31T22:00:00+00:00" and e.isoformat() == "2026-08-31T22:00:00+00:00"
    assert month_window(2026, 12)[1].isoformat() == "2026-12-31T23:00:00+00:00", "December roll-over (CET)"
    assert previous_month(date(2026, 1, 15)) == (2025, 12) and previous_of(2026, 1) == (2025, 12)
    assert parse_month("2026-08") == (2026, 8) and month_label(2026, 8) == "August 2026"
    now = datetime(2026, 9, 28, 8, 37, tzinfo=timezone.utc)          # a last-Monday run, 10:37 local
    s, e = report_window(2026, 9, now)
    assert e.isoformat() == "2026-09-28T08:00:00+00:00", e.isoformat()
    assert period_label(s, e) == "1 to 28 September 2026" and days_covered(s, e) == 28
    s, e = report_window(2026, 8, now)
    assert e == month_window(2026, 8)[1] and days_covered(s, e) == 31, "finished month must be full"
    return "UTC boundaries correct incl. DST, year roll-over and running-month cap"


# --------------------------------------------------------------------------- #
# 4-5: QuantumLeap, cleaning                                                   #
# --------------------------------------------------------------------------- #

@step("4 data/quantumleap.py")
def check_quantumleap(year: int, month: int):
    from data import quantumleap
    from data.timewindow import month_window
    ents = quantumleap.list_entities()
    assert len(ents) >= 50, f"only {len(ents)} entities listed"
    for t in ("Parking", "Weather", "AirQuality", "Traffic"):
        assert t in set(ents["entityType"]), f"type {t} missing"
    start, end = month_window(year, month)
    raw = quantumleap.fetch_history("ParkingSpot:Ulrichshaus", start, end)
    assert len(raw) > 500 and str(raw.index.tz) == "UTC" and "freeSpots" in raw.columns
    assert raw.index.min() >= start and raw.index.max() < end, "rows outside the window"
    _cache["raw"] = raw
    return f"{len(ents)} entities, {ents['entityType'].nunique()} types; Ulrichshaus {len(raw)} raw rows"


@step("5 data/cleaning.py")
def check_cleaning(year: int, month: int):
    import pandas as pd
    from data import cleaning
    from data.timewindow import month_window
    raw = _cache.get("raw")
    assert raw is not None, "step 4 did not produce data"
    start, end = month_window(year, month)
    ce = cleaning.clean("ParkingSpot:Ulrichshaus", raw, start, end)
    assert 0 < ce.rows < ce.raw_rows, f"dedupe did nothing: {ce.raw_rows} -> {ce.rows}"
    assert str(ce.readings.index.tz) == "Europe/Berlin"
    days = pd.Timestamp(year=year, month=month, day=1).days_in_month
    assert len(ce.hourly) == days * 24, f"hourly grid has {len(ce.hourly)} rows"
    assert 50 <= ce.coverage["coverage_pct"] <= 100
    return (f"{ce.raw_rows} raw -> {ce.rows} rows; grid {len(ce.hourly)} h; "
            f"coverage {ce.coverage['coverage_pct']}%, longest gap {ce.coverage['longest_gap_h']} h")


# --------------------------------------------------------------------------- #
# 6: preparation modules -> RhythmSeries                                       #
# --------------------------------------------------------------------------- #

def _check_series(series: dict, lo: float, hi: float, label: str):
    from Statistical_models.rhythm import RhythmSeries
    for k, s in series.items():
        assert isinstance(s, RhythmSeries), f"{label} {k}: not a RhythmSeries"
        assert s.usable or s.exclude_reason, f"{label} {k}: excluded without a reason"
        v = s.values.dropna()
        if s.usable:
            assert len(v) > 0 and lo <= v.min() and v.max() <= hi, f"{label} {k}: values outside [{lo}, {hi}]"


@step("6a data/orion.py")
def check_orion():
    from data import orion
    lots = orion.list_entities("Parking")
    assert len(lots) >= 5 and any(isinstance(e.get("totalSpots"), (int, float)) for e in lots)
    streets = orion.list_entities("Traffic")
    assert len(streets) > 1000, f"Orion paging broken: only {len(streets)} Traffic entities"
    return f"{len(lots)} lots with totalSpots; {len(streets)} street segments (paged)"


@step("6b data/parking.py")
def check_parking(year: int, month: int):
    from data.parking import prepare_parking
    from data.timewindow import month_window
    lots = prepare_parking(*month_window(year, month))
    assert len(lots) >= 5, f"only {len(lots)} lots"
    usable = [k for k, s in lots.items() if s.usable]
    assert len(usable) >= 3, f"only {len(usable)} usable lots"
    _check_series(lots, 0, 100, "parking")
    assert all(s.meta["capacity"] > 0 for s in lots.values() if s.usable)
    _cache["parking"] = lots
    return f"{len(usable)}/{len(lots)} usable ({', '.join(usable)})"


@step("6c data/traffic.py")
def check_traffic(year: int, month: int):
    from data.traffic import prepare_traffic
    from data.timewindow import month_window
    sensors = prepare_traffic(*month_window(year, month))
    assert len(sensors) >= 20, f"only {len(sensors)} traffic sensors"
    usable = [k for k, s in sensors.items() if s.usable]
    assert len(usable) >= 5, f"only {len(usable)} usable sensors"
    _check_series(sensors, 0, 150, "traffic")
    with_limit = sum(1 for s in sensors.values() if s.meta.get("speed_limit_kmh"))
    _cache["traffic"] = sensors
    return f"{len(usable)}/{len(sensors)} usable; {with_limit} with a speed limit from Orion"


@step("6d data/air_quality.py")
def check_air(year: int, month: int):
    from data.air_quality import prepare_air_quality
    from data.timewindow import month_window
    fam = prepare_air_quality(*month_window(year, month))
    assert set(fam) == {"no2", "o3", "pm10", "pm25"}, list(fam)
    no2_usable = [k for k, s in fam["no2"].items() if s.usable]
    assert len(no2_usable) >= 2, f"only {len(no2_usable)} usable NO2 stations"
    for p in fam:
        _check_series(fam[p], 0, 1000, f"air {p}")
    stale = [f"{p}/{k}" for p in fam for k, s in fam[p].items() if s.exclude_reason and "stale" in s.exclude_reason]
    _cache["air"] = fam
    return f"NO2 usable {no2_usable}; O3 {list(fam['o3'])}; stale excluded: {len(stale)}"


# --------------------------------------------------------------------------- #
# 7-8: the rhythm model on parking                                             #
# --------------------------------------------------------------------------- #

@step("7 Statistical_models/rhythm.py analyse")
def check_rhythm(year: int, month: int):
    from Statistical_models.rhythm import analyse_rhythm
    lots = _cache.get("parking")
    assert lots is not None, "step 6b did not produce lots"
    facts = analyse_rhythm(lots, year=year, month=month, variable="occupancy", unit="%", delta_unit="pp")
    assert len(facts["series"]) >= 3, f"only {len(facts['series'])} modelled"
    for k, f in facts["series"].items():
        assert 0 <= f["peak_hour"] <= 23 and 0 <= f["trough_hour"] <= 23, k
        assert len(f["profile_weekday"]) == 24 and len(f["profile_weekend"]) == 24, k
        assert 0 <= f["mean"] <= 100 and f["capacity"] > 0, k
        we = f["weekend_effect"]
        assert we is None or set(we) == {"delta", "ci95", "p_value", "significant"}, k
    assert facts["pooled"] is None or "weekend_effect_differs_between_series" in facts["pooled"]
    json.dumps(facts)
    peaks = ", ".join(f"{k} {v['peak_hour']:02d}:00" for k, v in facts["series"].items())
    return f"{len(facts['series'])} lots; peaks {peaks}"


@step("8 Statistical_models/rhythm.py compare")
def check_compare(year: int, month: int):
    from data.parking import prepare_parking
    from data.timewindow import month_window, previous_of
    from Statistical_models.rhythm import compare_rhythm
    cur = _cache.get("parking")
    assert cur is not None, "step 6b did not produce lots"
    py, pm = previous_of(year, month)
    prev = prepare_parking(*month_window(py, pm))
    facts = compare_rhythm(prev, cur, (py, pm), (year, month), variable="occupancy", unit="%", delta_unit="pp")
    assert facts["previous_month"] == f"{py}-{pm:02d}" and len(facts["series"]) >= 2
    for k, v in facts["series"].items():
        s = v["adjusted_shift"]
        assert s["ci95"][0] <= s["delta"] <= s["ci95"][1], f"{k}: shift outside its own CI"
        m = v["mean"]
        assert abs((m["current"] - m["previous"]) - m["change"]) < 0.11, f"{k}: change inconsistent"
    sm = facts["summary"]
    assert set(sm["significantly_higher"]) | set(sm["significantly_lower"]) | set(sm["no_significant_change"]) == set(facts["series"])
    json.dumps(facts)
    return f"{len(facts['series'])} lots vs {py}-{pm:02d}; higher {sm['significantly_higher'] or '-'}, lower {sm['significantly_lower'] or '-'}"


# --------------------------------------------------------------------------- #
# 9: facts.json                                                                #
# --------------------------------------------------------------------------- #

@step("9 report/facts.py")
def check_facts(year: int, month: int):
    from report.facts import build_facts, load_facts, save_facts, slim_facts
    facts = build_facts(year, month)
    for section in ("report", "highlights", "families", "glossary"):
        assert section in facts, f"section {section} missing"
    fams = facts["families"]
    for key in ("parking", "traffic", "air_no2", "air_o3", "air_pm10", "air_pm25"):
        assert key in fams, f"family {key} missing"
        for part in ("patterns", "comparison", "quality", "unit", "delta_unit", "higher_means"):
            assert part in fams[key], f"{key}.{part} missing"
    assert "thresholds" in fams["air_no2"]
    assert len(fams["parking"]["patterns"]["series"]) >= 3 and len(fams["traffic"]["patterns"]["series"]) >= 5
    assert isinstance(facts["highlights"], list)
    path = save_facts(facts)
    full_kb = path.stat().st_size / 1024
    slim_kb = len(json.dumps(slim_facts(facts))) / 1024
    assert slim_kb < full_kb and slim_kb < 250, f"slim facts {slim_kb:.0f} KB, too big for a prompt"
    assert load_facts(facts["report"]["month"])["report"]["month"] == facts["report"]["month"]
    return (f"{len(fams)} families, {len(facts['highlights'])} highlights; "
            f"{full_kb:.0f} KB full / {slim_kb:.0f} KB slim at {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# 10-11: LLM steps (opt-in with --llm)                                         #
# --------------------------------------------------------------------------- #

@step("10 report/writer.py (--llm)")
def check_writer(month: str):
    import config
    from report.facts import load_facts
    from report.verifier import untraced_numbers
    from report.writer import save_draft, write_report
    assert config.llm_configured(), "OPENAI_API_KEY missing"
    facts = load_facts(month)
    text = write_report(facts)
    assert text.startswith("# Imidillo report"), "draft must start with the title line"
    for heading in ("## Highlights", "## Parking", "## Traffic", "## Air quality", "## Data quality"):
        assert heading in text, f"missing section {heading!r}"
    assert "p-value" not in text.lower() and "p =" not in text, "draft must not quote p-values"
    words = len(text.split())
    assert 300 <= words <= 1100, f"{words} words, outside 300-1100"
    unknown = untraced_numbers(text, facts)
    assert not unknown, f"numbers not present in facts: {unknown}"
    path = save_draft(text, month)
    return f"{words} words, every number traced to facts, saved {path.relative_to(ROOT)}"


@step("11 report/verifier.py (--llm)")
def check_verifier(month: str):
    import config
    from report.facts import load_facts
    from report.verifier import INJECTED, untraced_numbers, verify
    facts = load_facts(month)
    draft_path = config.OUTPUT_DIR / month / "draft.md"
    assert draft_path.exists(), "no draft.md; run the writer step first"
    draft = draft_path.read_text(encoding="utf-8")
    poisoned = draft.replace("## Data quality", "\n".join(INJECTED) + "\n\n## Data quality")
    assert poisoned != draft, "could not inject"
    v = verify(facts, poisoned)
    assert v.items, "verifier returned no items"
    flagged = " ".join(i["claim"] for i in v.unsupported)
    assert "15:00" in flagged or "81.3" in flagged, "wrong peak hour / number NOT caught"
    assert "fair" in flagged.lower() or "because" in flagged.lower(), "invented cause NOT caught"
    assert "81.3" in untraced_numbers(poisoned, facts), "deterministic number check missed 81.3"
    return f"{v.count('supported')} supported, {v.count('unsupported')} unsupported; both injected fabrications caught"


# --------------------------------------------------------------------------- #
# 12-14: render, mailer, job                                                   #
# --------------------------------------------------------------------------- #

@step("12 report/render.py")
def check_render(month: str):
    import config
    from report.facts import load_facts
    from report.render import render, save_rendered
    facts = load_facts(month)
    md_path = config.OUTPUT_DIR / month / "report.md"
    if md_path.exists():
        report_md, source = md_path.read_text(encoding="utf-8"), "report.md"
    else:
        r = facts["report"]
        report_md, source = (f"# Imidillo report, {r['month_label']}\n\nStub.\n\n## Highlights\n- h\n\n"
                             f"## Parking\n- p\n\n## Traffic\n- t\n\n## Air quality\n- a\n\n"
                             f"## Data quality\n- q\n\nImidillo\n"), "stub"
    rendered = render(facts, report_md, with_pdf=True)
    assert rendered.subject.startswith("Imidillo report, ")
    cids = [cid for cid, _, _ in rendered.inline_images]
    assert "chart-parking" in cids and "chart-traffic" in cids, f"charts missing: {cids}"
    for cid in cids:
        assert f"cid:{cid}" in rendered.html, f"{cid} embedded but not referenced"
    assert rendered.html.count("<h1") == 1, "title must appear exactly once"
    assert "cid:chart-parking" in rendered.html.split("Traffic")[0], "parking chart must precede the traffic section"
    assert rendered.attachments and rendered.attachments[0][1][:4] == b"%PDF", "PDF attachment missing"
    path = save_rendered(rendered, month)
    kb = lambda b: f"{len(b) / 1024:.0f} KB"
    return f"from {source}; images {[(c, kb(b)) for c, b, _ in rendered.inline_images]}; pdf {kb(rendered.attachments[0][1])}"


@step("13 mailer.py (build only, no send)")
def check_mailer(month: str):
    import config
    from mailer import build_message, mailer_available
    from report.facts import load_facts
    from report.render import render
    assert mailer_available(), "SMTP not configured"
    facts = load_facts(month)
    md_path = config.OUTPUT_DIR / month / "report.md"
    report_md = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Imidillo report, x\n\nstub\n"
    rendered = render(facts, report_md, with_pdf=True)
    to = ["smoke-test@example.invalid"]
    msg = build_message(rendered.subject, rendered.html, rendered.text, to, rendered.attachments, rendered.inline_images)
    assert msg["To"] == to[0] and config.SMTP_USERNAME in msg["From"]
    types = [p.get_content_type() for p in msg.walk()]
    assert msg.get_content_type() == "multipart/mixed" and "text/plain" in types and "text/html" in types
    assert "multipart/related" in types, "inline images must sit in a multipart/related HTML part"
    cids = {p["Content-ID"].strip("<>") for p in msg.walk() if p["Content-ID"]}
    assert {c for c, _, _ in rendered.inline_images} <= cids, f"missing CIDs: {cids}"
    assert sum(1 for p in msg.walk() if p.get_content_type() == "application/pdf") == 1
    size_kb = len(msg.as_bytes()) / 1024
    assert size_kb < 5000, f"message is {size_kb:.0f} KB"
    return f"multipart/mixed, {len(cids)} inline images, 1 PDF, {size_kb:.0f} KB on the wire"


@step("14 jobs/monthly_report.py (exit codes)")
def check_job():
    import config
    from jobs import monthly_report
    saved = config.OPENAI_API_KEY
    try:
        config.OPENAI_API_KEY = ""
        assert monthly_report.run(2026, 8, dry_run=True) == 2, "missing key must give exit 2"
    finally:
        config.OPENAI_API_KEY = saved
    called = {}
    def boom(*a, **k):
        called["facts"] = True
        raise RuntimeError("simulated QuantumLeap outage")
    original = monthly_report.build_facts
    monthly_report.build_facts = boom
    try:
        assert monthly_report.run(2026, 8, dry_run=True) == 1, "failure must give exit 1"
    finally:
        monthly_report.build_facts = original
    assert called.get("facts"), "run() never reached the facts stage"
    return "not configured -> 2, failure -> 1; full dry run is `python -m jobs.monthly_report --dry-run`"


# --------------------------------------------------------------------------- #

def main() -> int:
    from data.timewindow import parse_month, previous_month
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_llm = "--llm" in sys.argv
    year, month = parse_month(args[0]) if args else previous_month()
    ym = f"{year}-{month:02d}"
    print(f"Sleeping Imidillo smoke test, report month {ym}"
          f"{' (with LLM steps)' if run_llm else ' (LLM steps skipped; add --llm)'}\n")

    check_deploy()
    check_config()
    check_timewindow()
    check_quantumleap(year, month)
    check_cleaning(year, month)
    check_orion()
    check_parking(year, month)
    check_traffic(year, month)
    check_air(year, month)
    check_rhythm(year, month)
    check_compare(year, month)
    check_facts(year, month)
    if run_llm:
        check_writer(ym)
        check_verifier(ym)
    check_render(ym)
    check_mailer(ym)
    check_job()

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)} passed, {len(failed)} failed"
          + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
