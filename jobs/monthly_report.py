"""
Monthly report job: the one entry point the systemd timer runs.

    timer (last Monday, 10:00) -> python -m jobs.monthly_report
              1. facts      report/facts.py       (QuantumLeap, Orion, statistics)
              2. write      report/writer.py      (LLM)
              3. verify     report/verifier.py    (LLM, revise loop, cut what fails)
              4. render     report/render.py      (HTML, charts, PDF)
              5. send       mailer.py             (to REPORT_EMAIL_TO)

Which month: the CURRENT calendar month, covered up to the last full hour
before the run (the timer fires before the month ends), compared with the
previous, complete month. Pass a month explicitly to rebuild an older one.

Everything is written to output/<month>/ along the way, so a failed run can
be inspected and the LLM steps re-run by hand.

Nothing is sent unless the verifier approved the text. Exit codes:
    0  report sent (or --dry-run completed)
    2  not configured (missing .env settings): skipped, not an error
    1  failure; the reason is in the log, nothing was sent

Usage:
    python -m jobs.monthly_report                          # current month, to REPORT_EMAIL_TO
    python -m jobs.monthly_report --dry-run                # everything except sending
    python -m jobs.monthly_report 2026-08                  # rebuild a past month
    python -m jobs.monthly_report --to me@ovgu.de          # send to these instead of the list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import config
from data.orion import OrionError
from data.quantumleap import QuantumLeapError
from data.timewindow import current_month, month_label, parse_month
from mailer import MailerError, send_rendered
from report.facts import build_facts, save_facts
from report.llm import LLMError
from report.render import render, save_rendered
from report.verifier import verify_and_fix
from report.writer import save_draft, write_report


def _log(stage: str, msg: str, t0: float | None = None) -> None:
    took = f" ({time.perf_counter() - t0:.0f}s)" if t0 is not None else ""
    print(f"[JOB] {stage:7s} {msg}{took}", flush=True)


def run(year: int, month: int, dry_run: bool = False, to: list[str] | None = None) -> int:
    label = month_label(year, month)
    ym = f"{year}-{month:02d}"
    _log("start", f"report for {label}{' (dry run)' if dry_run else ''}")

    missing = []
    if not config.quantumleap_configured():
        missing.append("QL_BASE_URL")
    if not config.llm_configured():
        missing.append("OPENAI_API_KEY")
    if not dry_run and not config.smtp_configured():
        missing.append("SMTP_* / REPORT_EMAIL_TO")
    if missing:
        _log("skip", f"not configured: {', '.join(missing)} (see .env.example)")
        return 2

    job_t0 = time.perf_counter()
    try:
        # 1. facts
        t0 = time.perf_counter()
        facts = build_facts(year, month)
        save_facts(facts)
        r = facts["report"]
        usable = ", ".join(f"{k} {len(f['quality']['usable'])}/{f['quality']['total']}"
                           for k, f in facts["families"].items())
        _log("facts", f"{r['period_label']} ({r['days_covered']}/{r['days_in_month']} days); "
                      f"usable sensors: {usable}; {len(facts['highlights'])} highlights", t0)

        # 2. write
        t0 = time.perf_counter()
        draft = write_report(facts)
        save_draft(draft, ym)
        _log("write", f"{len(draft.split())} words", t0)

        # 3. verify (revise loop inside; cuts what still fails)
        t0 = time.perf_counter()
        text, history = verify_and_fix(facts, draft)
        final = history[-1]
        out_dir = config.OUTPUT_DIR / ym
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        (out_dir / "verification.json").write_text(
            json.dumps([v.to_dict() for v in history], indent=2, ensure_ascii=False), encoding="utf-8")
        approved = final.passed or bool(final.stripped)
        _log("verify", f"{final.summary()} -> {'approved' if approved else 'NOT approved'}", t0)
        if not approved:
            _log("abort", "verifier did not approve the text; nothing sent")
            return 1

        # 4. render
        t0 = time.perf_counter()
        rendered = render(facts, text)
        save_rendered(rendered, ym)
        _log("render", f"{len(rendered.inline_images)} images, {len(rendered.attachments)} attachment(s)", t0)

        # 5. send
        if dry_run:
            _log("done", f"dry run complete, see {out_dir}", job_t0)
            return 0
        t0 = time.perf_counter()
        delivered = send_rendered(rendered, to=to)
        if not delivered:
            _log("abort", "no recipient accepted the message")
            return 1
        _log("sent", f"to {', '.join(delivered)}", t0)
        _log("done", f"{label} report delivered", job_t0)
        return 0

    except (QuantumLeapError, OrionError, LLMError, MailerError) as e:
        _log("error", f"{type(e).__name__}: {e}")
        return 1
    except Exception as e:  # noqa: BLE001  anything else: log the trace, fail loudly
        _log("error", f"unexpected {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, verify and email the monthly report.")
    parser.add_argument("month", nargs="?", help="YYYY-MM; default: the current calendar month")
    parser.add_argument("--dry-run", action="store_true", help="do everything except sending")
    parser.add_argument("--to", help="recipient(s), comma-separated, instead of REPORT_EMAIL_TO")
    args = parser.parse_args()
    year, month = parse_month(args.month) if args.month else current_month()
    to = [a.strip() for a in args.to.split(",") if a.strip()] if args.to else None
    return run(year, month, dry_run=args.dry_run, to=to)


if __name__ == "__main__":
    sys.exit(main())
