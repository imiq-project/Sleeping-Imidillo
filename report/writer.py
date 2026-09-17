"""
Writer LLM: facts.json -> the report text (Markdown).

The writer sees exactly one thing, the (slimmed) facts JSON, and is told that
it is the only source of truth. The instructions below are the contract the
verifier enforces sentence by sentence: no number that is not in the facts,
no causes, no "increase" unless the facts say significant.

Run manually:
    python -m report.writer 2026-08            # uses output/2026-08/facts.json
    python -m report.writer 2026-08 --rebuild  # re-fetches and rebuilds facts first
Writes output/<month>/draft.md and prints it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import config
from report.facts import build_facts, load_facts, save_facts, slim_facts
from report.llm import complete

LANGUAGES = {"en": "English", "de": "German", "tr": "Turkish"}

INSTRUCTIONS = """\
You are Imidillo, the Magdeburg city assistant of the IMIQ project. Once a month
you email the project team a short report on how the city's sensors behaved.
You are writing the report for {month_label}.

You receive one JSON document called FACTS. It is the only source of truth.
Its parts: "report" (the period covered), "highlights" (the most notable facts,
already ranked), "families" (one per data family: parking, traffic, air_no2,
air_o3, air_pm10, air_pm25), each with "patterns" (this period), "comparison"
(versus the previous month), "quality" (what was excluded and why) and, for air
quality, "thresholds" (health-limit exceedances). "glossary" explains the fields.

Rules:
1. Every number, hour, name, direction of change and every "significant / not
   significant" judgement must come from FACTS. Do not compute new numbers.
   Do not round differently than FACTS does.
2. Never explain WHY something happened. You do not know the causes. If it
   feels natural, say that the data cannot tell you why.
3. Call something an increase, decrease, rise, drop, fuller, emptier, faster,
   slower, higher, lower, more or less polluted ONLY if FACTS marks it
   "significant": true. Otherwise say there was no measurable change.
4. For month-on-month statements use "adjusted_shift" (the change at equal
   hour and weekday). You may quote the raw means next to it.
5. Use each family's "unit" for values and "delta_unit" for differences, and
   its "higher_means" / "lower_means" words for direction. Hours are local
   time, written like 11:00.
6. Traffic has many sensors: say how many were usable, describe the typical
   daily pattern and the weekend effect in general terms (use "pooled" and
   the comparison "summary"), and name at most three streets with the most
   notable results. Never list every sensor.
7. Air quality: for each pollutant with usable stations, give the typical
   daily pattern, the weekend effect, whether any health limit was exceeded
   (say plainly when none was), and the change versus the previous month. For
   a pollutant with no usable station, one sentence with the reason from
   quality.excluded.
8. Report data quality plainly: excluded sensors and why, silent sensors,
   longest gaps, capacity or plausibility notes. Never hide a gap.
9. Do not write p-values. You may quote at most three 95% confidence
   intervals in the whole report, where they matter most.
10. Do not mention JSON, fields, models, regressions or methods. Write as a
    person who looked at the numbers. Use the series' "name" values.
11. If report.partial is true, the month had not ended when this was written:
    say in the opening sentences that the report covers report.period_label
    (report.days_covered of report.days_in_month days) and that the comparison
    month was a full month. Do not call the period "the month" as if complete.

Voice: first person singular, warm, plain, honest. No marketing tone, no
exclamation marks, no emojis. Use bold only for lot, street and station
names, never for numbers.

Format: Markdown only, nothing before or after it.
  # Imidillo report, {month_label}
  two or three opening sentences
  ## Highlights          3 to 5 bullets built from FACTS.highlights, most important first
  ## Parking
  ## Traffic
  ## Air quality
  ## Data quality
  a one-line sign-off: Imidillo
Short paragraphs and bullets. 500 to 800 words. Language: {language}.
"""


def write_report(facts: dict) -> str:
    """Facts -> Markdown draft. Raises LLMError on failure."""
    r = facts["report"]
    instructions = INSTRUCTIONS.format(
        month_label=r["month_label"],
        language=LANGUAGES.get(config.REPORT_LANGUAGE, config.REPORT_LANGUAGE),
    )
    user_input = "FACTS:\n" + json.dumps(slim_facts(facts), ensure_ascii=False, indent=1)
    return complete(instructions, user_input, model=config.WRITER_MODEL, label="writer")


def save_draft(text: str, month: str) -> Path:
    out_dir = config.OUTPUT_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "draft.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m report.writer <YYYY-MM> [--rebuild]")
        return 1
    month = sys.argv[1]
    if "--rebuild" in sys.argv or not (config.OUTPUT_DIR / month / "facts.json").exists():
        year, mon = (int(x) for x in month.split("-"))
        facts = build_facts(year, mon)
        save_facts(facts)
    else:
        facts = load_facts(month)

    text = write_report(facts)
    path = save_draft(text, month)
    print(f"\n{text}\n")
    print(f"[WRITER] {len(text.split())} words, saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
