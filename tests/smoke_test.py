"""
Smoke test: run every pipeline stage built so far, end to end, in one go.

Each step prints PASS or FAIL with a one-line reason. Network steps talk to
the real Orion / QuantumLeap; they need internet but no API key. The test
fetches one report month (default: 2026-08) and reuses the fetched data
across steps so it stays under ~30 seconds.

    python -m tests.smoke_test                  # previous calendar month
    python -m tests.smoke_test 2026-08          # a specific month
    python -m tests.smoke_test 2026-08 --llm    # also run the LLM steps (tokens, ~1 min)

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
            print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} {dt:5.1f}s  {note}")
            _results.append((name, ok, note))
            return out
        return run
    return wrap


# --------------------------------------------------------------------------- #
# Step 1: deploy files                                                         #
# --------------------------------------------------------------------------- #

@step("1 deploy/ unit files")
def check_deploy():
    d = ROOT / "deploy"
    expected = ["imidillo-report.service", "imidillo-report.timer", "imidillo-report-test.timer"]
    for name in expected:
        f = d / name
        assert f.exists(), f"missing {f.name}"
        raw = f.read_bytes()
        assert b"\r" not in raw, f"{name} has Windows line endings (CR); systemd would choke"
    service = (d / "imidillo-report.service").read_text()
    assert "ExecStart=" in service and "jobs.monthly_report" in service, "service ExecStart wrong"
    assert "SuccessExitStatus=2" in service, "service lacks SuccessExitStatus=2"
    timer = (d / "imidillo-report.timer").read_text()
    assert "OnCalendar=Mon *-*~07/1" in timer, "timer OnCalendar not 'last Monday'"
    assert "Persistent=true" in timer, "timer lacks Persistent=true"
    test_timer = (d / "imidillo-report-test.timer").read_text()
    assert "Unit=imidillo-report.service" in test_timer, "test timer must point at the service"
    return "3 files, LF endings, last-Monday schedule"


# --------------------------------------------------------------------------- #
# Step 2: config                                                               #
# --------------------------------------------------------------------------- #

@step("2 config.py")
def check_config():
    import config
    assert (ROOT / ".env").exists(), ".env missing"
    assert (ROOT / ".gitignore").exists() and ".env" in (ROOT / ".gitignore").read_text(), \
        ".gitignore must list .env"
    assert config.LOCAL_TZ == "Europe/Berlin", config.LOCAL_TZ
    assert config.QL_BASE_URL.startswith("https://"), config.QL_BASE_URL
    assert config.ORION_BASE_URL.startswith("https://"), config.ORION_BASE_URL
    assert config.quantumleap_configured(), "quantumleap_configured() is False"
    assert "FIWARE_API_KEY" not in config.missing(), "FIWARE_API_KEY must not be required"
    later = [n for n in config.missing()]
    return (f"QL ready; SMTP ready={config.smtp_configured()}; LLM ready={config.llm_configured()}"
            + (f"; still missing for the full job: {', '.join(later)}" if later else ""))


# --------------------------------------------------------------------------- #
# Step 3: time window                                                          #
# --------------------------------------------------------------------------- #

@step("3 data/timewindow.py")
def check_timewindow():
    from data.timewindow import month_label, month_window, parse_month, previous_month
    s, e = month_window(2026, 8)
    assert s.tzinfo == timezone.utc and e.tzinfo == timezone.utc, "window must be UTC"
    assert s.isoformat() == "2026-07-31T22:00:00+00:00", s.isoformat()   # CEST midnight = 22:00 UTC
    assert e.isoformat() == "2026-08-31T22:00:00+00:00", e.isoformat()
    s, e = month_window(2026, 12)
    assert e.isoformat() == "2026-12-31T23:00:00+00:00", "December end must roll into next year (CET)"
    assert previous_month(date(2026, 1, 15)) == (2025, 12)
    assert parse_month("2026-08") == (2026, 8)
    assert month_label(2026, 8) == "August 2026"
    # running month: capped at "now" floored to the hour; finished month: full
    from datetime import datetime
    from data.timewindow import days_covered, period_label, report_window
    now = datetime(2026, 9, 28, 8, 37, tzinfo=timezone.utc)      # a last-Monday run, 10:37 local
    s, e = report_window(2026, 9, now)
    assert e.isoformat() == "2026-09-28T08:00:00+00:00", e.isoformat()
    assert period_label(s, e) == "1 to 28 September 2026", period_label(s, e)
    assert days_covered(s, e) == 28, days_covered(s, e)
    s, e = report_window(2026, 8, now)
    assert e == month_window(2026, 8)[1] and days_covered(s, e) == 31, "finished month must be full"
    return "UTC boundaries correct incl. DST, year roll-over and running-month cap"


# --------------------------------------------------------------------------- #
# Step 4: QuantumLeap client                                                   #
# --------------------------------------------------------------------------- #

@step("4 data/quantumleap.py")
def check_quantumleap(year: int, month: int):
    from data import quantumleap
    from data.timewindow import month_window
    ents = quantumleap.list_entities()
    assert len(ents) >= 50, f"only {len(ents)} entities listed"
    types = set(ents["entityType"])
    for t in ("Parking", "Weather", "AirQuality"):
        assert t in types, f"type {t} missing from QuantumLeap inventory"
    start, end = month_window(year, month)
    raw = quantumleap.fetch_history("ParkingSpot:Ulrichshaus", start, end)
    assert len(raw) > 500, f"only {len(raw)} raw rows for Ulrichshaus"
    assert str(raw.index.tz) == "UTC", f"raw index tz is {raw.index.tz}, expected UTC"
    assert "freeSpots" in raw.columns, raw.columns.tolist()
    assert raw.index.min() >= start and raw.index.max() < end, "rows outside the window"
    check_quantumleap.raw = raw
    return f"{len(ents)} entities, {len(types)} types; Ulrichshaus {len(raw)} raw rows"


# --------------------------------------------------------------------------- #
# Step 5: cleaning                                                             #
# --------------------------------------------------------------------------- #

@step("5 data/cleaning.py")
def check_cleaning(year: int, month: int):
    import pandas as pd
    from data import cleaning
    from data.timewindow import month_window
    raw = getattr(check_quantumleap, "raw", None)
    assert raw is not None, "step 4 did not produce data"
    start, end = month_window(year, month)
    ce = cleaning.clean("ParkingSpot:Ulrichshaus", raw, start, end)
    assert 0 < ce.rows < ce.raw_rows, f"dedupe did nothing: {ce.raw_rows} -> {ce.rows}"
    assert str(ce.readings.index.tz) == "Europe/Berlin", f"readings tz {ce.readings.index.tz}"
    days = pd.Timestamp(year=year, month=month, day=1).days_in_month
    assert len(ce.hourly) == days * 24, f"hourly grid has {len(ce.hourly)} rows, expected {days * 24}"
    cov = ce.coverage
    assert 50 <= cov["coverage_pct"] <= 100, cov
    assert cov["hours_with_data"] == int(ce.hourly["freeSpots"].notna().sum())
    return (f"{ce.raw_rows} raw -> {ce.rows} rows; grid {len(ce.hourly)} h; "
            f"coverage {cov['coverage_pct']}%, longest gap {cov['longest_gap_h']} h")


# --------------------------------------------------------------------------- #
# Step 6: Orion + parking preparation                                          #
# --------------------------------------------------------------------------- #

@step("6a data/orion.py")
def check_orion():
    from data import orion
    ents = orion.list_entities("Parking")
    assert len(ents) >= 5, f"only {len(ents)} Parking entities in Orion"
    with_total = [e for e in ents if isinstance(e.get("totalSpots"), (int, float))]
    assert with_total, "no entity carries totalSpots"
    return f"{len(ents)} lots, {len(with_total)} with totalSpots"


@step("6b data/parking.py")
def check_parking(year: int, month: int):
    from data.parking import prepare_parking
    from data.timewindow import month_window
    start, end = month_window(year, month)
    lots = prepare_parking(start, end)
    assert len(lots) >= 5, f"only {len(lots)} lots prepared"
    usable = {k: v for k, v in lots.items() if v.usable}
    assert len(usable) >= 3, f"only {len(usable)} usable lots: {list(usable)}"
    for k, lot in usable.items():
        s = lot.occ_pct.dropna()
        assert len(s) > 0 and s.min() >= 0 and s.max() <= 100, f"{k}: occ_pct out of [0,100]"
        assert lot.capacity > 0, f"{k}: capacity {lot.capacity}"
    for k, lot in lots.items():
        assert lot.usable or lot.exclude_reason, f"{k}: excluded without a reason"
    check_parking.lots = lots
    notes = [k for k, v in lots.items() if "totalSpots from Orion" not in v.capacity_note]
    return f"{len(usable)}/{len(lots)} usable ({', '.join(usable)}); capacity notes: {notes or 'none'}"


# --------------------------------------------------------------------------- #
# Step 7: parking model -> facts                                               #
# --------------------------------------------------------------------------- #

@step("7 Statistical_models/parking_hourly_regression.py")
def check_model(year: int, month: int):
    from Statistical_models.parking_hourly_regression import analyse_parking
    lots = getattr(check_parking, "lots", None)
    assert lots is not None, "step 6b did not produce lots"
    facts = analyse_parking(lots, year, month)
    assert facts["section"] == "parking_patterns"
    assert facts["month"] == f"{year}-{month:02d}"
    assert len(facts["lots"]) >= 3, f"only {len(facts['lots'])} lots modelled"
    for k, f in facts["lots"].items():
        assert 0 <= f["peak_hour"] <= 23 and 0 <= f["trough_hour"] <= 23, k
        assert len(f["profile_weekday_pct"]) == 24 and len(f["profile_weekend_pct"]) == 24, k
        assert 0 <= f["mean_occ_pct"] <= 100, k
        we = f["weekend_effect"]
        assert we is None or set(we) == {"pp", "ci95", "p_value", "significant"}, k
    json.dumps(facts)  # must be serialisable
    assert facts["pooled"] is None or "weekend_effect_differs_by_lot" in facts["pooled"]
    peaks = ", ".join(f"{k} {v['peak_hour']:02d}:00" for k, v in facts["lots"].items())
    return f"{len(facts['lots'])} lots modelled; peaks {peaks}"


# --------------------------------------------------------------------------- #
# Step 8: comparison with the previous month                                   #
# --------------------------------------------------------------------------- #

@step("8 Statistical_models/parking_comparison.py")
def check_comparison(year: int, month: int):
    from data.parking import prepare_parking
    from data.timewindow import month_window, previous_of
    from Statistical_models.parking_comparison import compare_parking
    cur_lots = getattr(check_parking, "lots", None)
    assert cur_lots is not None, "step 6b did not produce lots"
    py, pm = previous_of(year, month)
    assert previous_of(2026, 1) == (2025, 12)
    prev_lots = prepare_parking(*month_window(py, pm))
    facts = compare_parking(prev_lots, cur_lots, (py, pm), (year, month))
    assert facts["section"] == "parking_comparison"
    assert facts["previous_month"] == f"{py}-{pm:02d}"
    assert len(facts["lots"]) >= 2, f"only {len(facts['lots'])} lots compared"
    for k, v in facts["lots"].items():
        s = v["adjusted_shift"]
        assert s["ci95"][0] <= s["pp"] <= s["ci95"][1], f"{k}: shift outside its own CI"
        m = v["mean_occ_pct"]
        assert abs((m["current"] - m["previous"]) - m["change_pp"]) < 0.11, f"{k}: change_pp inconsistent"
        assert 0 <= v["peak"]["current"]["hour"] <= 23
    sm = facts["summary"]
    assert sm["lots_compared"] == len(facts["lots"])
    assert set(sm["significantly_fuller"]) | set(sm["significantly_emptier"]) | set(sm["no_significant_change"]) == set(facts["lots"])
    json.dumps(facts)
    return (f"{len(facts['lots'])} lots vs {py}-{pm:02d}; fuller {sm['significantly_fuller'] or '-'}, "
            f"emptier {sm['significantly_emptier'] or '-'}")


# --------------------------------------------------------------------------- #
# Step 9: facts.json assembly                                                  #
# --------------------------------------------------------------------------- #

@step("9 report/facts.py")
def check_facts(year: int, month: int):
    from report.facts import build_facts, load_facts, save_facts
    facts = build_facts(year, month)
    for section in ("report", "data_quality", "parking_patterns", "parking_comparison", "glossary"):
        assert section in facts, f"section {section} missing"
    assert facts["report"]["month"] == f"{year}-{month:02d}"
    q = facts["data_quality"]["parking"]
    assert q["lots_usable"] == len(q["usable"]) and q["lots_total"] == len(q["usable"]) + len(q["excluded"])
    assert set(facts["parking_patterns"]["lots"]) == set(q["usable"]) - set(facts["parking_patterns"]["skipped"])
    path = save_facts(facts)
    assert path.exists() and path.name == "facts.json"
    size_kb = path.stat().st_size / 1024
    assert size_kb < 200, f"facts.json is {size_kb:.0f} KB, too big for an LLM prompt"
    reloaded = load_facts(facts["report"]["month"])
    assert reloaded["report"]["month"] == facts["report"]["month"], "load_facts round-trip failed"
    return f"5 sections, {q['lots_usable']}/{q['lots_total']} lots, {size_kb:.1f} KB at {path.relative_to(ROOT)}"


# --------------------------------------------------------------------------- #
# Step 10: writer LLM (opt-in with --llm: costs tokens and ~1 minute)          #
# --------------------------------------------------------------------------- #

@step("10 report/writer.py (--llm)")
def check_writer(month: str):
    import re
    import config
    from report.facts import load_facts
    from report.writer import save_draft, write_report
    assert config.llm_configured(), "OPENAI_API_KEY missing; cannot test the writer"
    facts = load_facts(month)
    text = write_report(facts)
    assert text.startswith("# Imidillo report"), "draft must start with the title line"
    for heading in ("## Parking this month", "## Compared with", "## Data quality"):
        assert heading in text, f"missing section {heading!r}"
    assert "p-value" not in text.lower() and "p =" not in text, "draft must not quote p-values"
    words = len(text.split())
    assert 200 <= words <= 600, f"{words} words, outside 200-600"
    from report.verifier import untraced_numbers
    unknown = untraced_numbers(text, facts)
    assert not unknown, f"numbers not present in facts: {unknown}"
    path = save_draft(text, month)
    return f"{words} words, every number traced to facts, saved {path.relative_to(ROOT)}"


# --------------------------------------------------------------------------- #
# Step 11: verifier LLM must catch fabrications (opt-in with --llm)            #
# --------------------------------------------------------------------------- #

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
    assert poisoned != draft, "could not inject (no '## Data quality' heading in draft)"
    v = verify(facts, poisoned)
    assert v.items, "verifier returned no items"
    flagged = " ".join(i["claim"] for i in v.unsupported)
    assert "15:00" in flagged or "81.3" in flagged, "wrong peak hour / number NOT caught"
    assert "fair" in flagged.lower() or "because" in flagged.lower(), "invented cause NOT caught"
    assert untraced_numbers(poisoned, facts) == ["81.3"] or "81.3" in untraced_numbers(poisoned, facts), \
        "deterministic number check missed 81.3"
    return (f"{v.count('supported')} supported, {v.count('unsupported')} unsupported; "
            f"both injected fabrications caught")


# --------------------------------------------------------------------------- #
# Step 12: rendering (deterministic; uses report.md if present, else a stub)   #
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
        report_md, source = (f"# Imidillo report, {r['month_label']}\n\nStub text.\n\n"
                             f"## Parking this month\n- one\n\n## Compared with {r['previous_month_label']}\n"
                             f"- two\n\n## Data quality\n- three\n\nImidillo\n"), "stub"
    rendered = render(facts, report_md, with_pdf=True)
    assert rendered.subject.startswith("Imidillo report, "), rendered.subject
    cids = [cid for cid, _, _ in rendered.inline_images]
    assert "chart-profiles" in cids and "chart-comparison" in cids, f"charts missing: {cids}"
    for cid in cids:
        assert f"cid:{cid}" in rendered.html, f"{cid} embedded but not referenced in html"
    assert rendered.html.count("<h1") == 1, "title must appear exactly once (header only)"
    assert "cid:chart-profiles" in rendered.html.split("Compared with")[0], "profiles chart must precede the comparison section"
    assert rendered.text == report_md
    assert rendered.attachments and rendered.attachments[0][0].endswith(".pdf"), "PDF attachment missing"
    assert rendered.attachments[0][1][:4] == b"%PDF", "attachment is not a PDF"
    path = save_rendered(rendered, month)
    kb = lambda b: f"{len(b) / 1024:.0f} KB"
    return (f"from {source}; images {[(c, kb(b)) for c, b, _ in rendered.inline_images]}; "
            f"pdf {kb(rendered.attachments[0][1])}; preview {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Step 13: mailer builds a correct MIME message (nothing is sent)              #
# --------------------------------------------------------------------------- #

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
    msg = build_message(rendered.subject, rendered.html, rendered.text, to,
                        rendered.attachments, rendered.inline_images)
    assert msg["To"] == to[0] and config.SMTP_USERNAME in msg["From"], (msg["To"], msg["From"])
    assert msg.get_content_type() == "multipart/mixed", msg.get_content_type()
    parts = list(msg.walk())
    types = [p.get_content_type() for p in parts]
    assert "text/plain" in types and "text/html" in types, types
    assert "multipart/related" in types, "inline images must sit in a multipart/related HTML part"
    cids = {p["Content-ID"].strip("<>") for p in parts if p["Content-ID"]}
    assert {c for c, _, _ in rendered.inline_images} <= cids, f"missing CIDs: {cids}"
    pdfs = [p for p in parts if p.get_content_type() == "application/pdf"]
    assert len(pdfs) == 1 and pdfs[0].get_filename().endswith(".pdf"), "PDF attachment missing"
    size_kb = len(msg.as_bytes()) / 1024
    assert size_kb < 5000, f"message is {size_kb:.0f} KB, too large"
    return f"multipart/mixed, {len(cids)} inline images, 1 PDF, {size_kb:.0f} KB on the wire"


# --------------------------------------------------------------------------- #
# Step 14: the job's exit-code contract (no network, no LLM, nothing sent)     #
# --------------------------------------------------------------------------- #

@step("14 jobs/monthly_report.py (exit codes)")
def check_job():
    import config
    from jobs import monthly_report
    saved = config.OPENAI_API_KEY
    try:
        config.OPENAI_API_KEY = ""                      # not configured -> 2, and nothing else runs
        assert monthly_report.run(2026, 8, dry_run=True) == 2, "missing key must give exit 2"
    finally:
        config.OPENAI_API_KEY = saved
    called = {}
    def boom(*a, **k):
        called["facts"] = True
        raise RuntimeError("simulated QuantumLeap outage")
    original = monthly_report.build_facts
    monthly_report.build_facts = boom                   # any failure -> 1, nothing sent
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
    print(f"Sleeping Imidillo smoke test, report month {year}-{month:02d}"
          f"{' (with LLM steps)' if run_llm else ' (LLM steps skipped; add --llm)'}\n")

    check_deploy()
    check_config()
    check_timewindow()
    check_quantumleap(year, month)
    check_cleaning(year, month)
    check_orion()
    check_parking(year, month)
    check_model(year, month)
    check_comparison(year, month)
    check_facts(year, month)
    if run_llm:
        check_writer(f"{year}-{month:02d}")
        check_verifier(f"{year}-{month:02d}")
    check_render(f"{year}-{month:02d}")
    check_mailer(f"{year}-{month:02d}")
    check_job()

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(failed)} passed, {len(failed)} failed"
          + (f": {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
