"""
Experiment B: what does the statistics layer buy? Raw-data LLM vs. pipeline.

Three conditions, same model, same reasoning effort, same voice and format rules:
  raw             the writer gets the cleaned hourly data itself (the same hourly
                  series the statistics layer uses, for the report month and the
                  previous month) and must compute everything on its own
  pipeline_draft  the production writer: it gets only the facts.json
  pipeline_final  that draft after the production verify-and-fix loop
Only the difference "who computes the statistics" separates raw from
pipeline_draft: the data, cleaning, coverage rules, health limits, sections and
honesty rules are the same.

Scope: the Parking, EV charging and Air quality sections (traffic, 29 sensors,
would make the raw prompt too large; it is left out of all three conditions).

Scoring: every scored section of every report is checked against the TRUE
facts by the production verifier as a judge (fresh calls, temperature as in
production), giving supported / unsupported / unverifiable claim counts, plus
the deterministic number tracer. Experiment A estimates how reliable this judge
is. A judge from the same model family is a limitation, stated in the paper.

Run:
    python -m experiments.exp_b_raw_baseline 2026-08 2026-09
Writes output/experiments/exp_b/<run>/ (data CSVs, every report, judgements,
summary.json).
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

import config
from data import air_quality, ev_charging, parking
from data.timewindow import month_label, month_window, parse_month, period_label, previous_of
from report.facts import load_facts, slim_facts
from report.llm import complete
from report.verifier import _verify_section, split_sections, untraced_numbers, verify_and_fix
from report.writer import write_report

RUNS = 3
SCORED = ["Parking", "EV charging", "Air quality"]
RETRIES = 4


def _retry(fn, *args):
    """Experiment-level retry around a whole stage call: report.llm retries a
    few seconds apart, which is too short for a per-minute rate limit."""
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            if attempt == RETRIES:
                raise
            print(f"[EXP-B] {fn.__name__} failed ({type(e).__name__}: {str(e)[:120]}), retry {attempt} in 60 s",
                  flush=True)
            time.sleep(60)
LANG = {"en": "English", "de": "German", "tr": "Turkish"}

RAW_INSTRUCTIONS = """\
You are Imidillo, the Magdeburg city assistant of the IMIQ project. Once a month
you email the project team a short report on how the city's sensors behaved.
You are writing the report for {month_label}.

You receive DATA: hourly sensor values (local time, Europe/Berlin) for the report
month and for the whole previous month ({previous_month_label}). An empty cell
means no reading in that hour. Analyse the data yourself: averages, the typical
weekday daily pattern (busiest and quietest hour), weekend versus weekday
differences, changes versus the previous month at comparable hours and days,
health-limit exceedances, and data quality.

Health limits (µg/m³): NO2 EU hourly limit 200, WHO daily-mean guideline 25;
O3 EU 8-hour target 120, WHO 8-hour guideline 100; PM10 EU daily limit 50, WHO
daily guideline 45; PM2.5 WHO daily guideline 15.
Data-quality rules (the same as in the production pipeline): a sensor with
readings in fewer than 60 % of the hours is not reliable enough to describe. A
pollutant sensor (NO2, O3, PM10, PM2.5) whose value does not change from one hour
to the next in more than half of the hours is also unreliable. This rule does
NOT apply to parking or EV charging: EV values are reconstructed charging states,
and long runs of unchanged hours are normal there.

Rules:
1. Every number, hour, name and direction of change must be supported by DATA.
2. Never explain WHY something happened. You do not know the causes.
3. Call something an increase, decrease, fuller, emptier, busier, more or less
   polluted only if the data clearly supports it; otherwise say there was no
   measurable change.
4. Hours are local time, written like 11:00. Percentages with a % sign;
   differences in "percentage points".
5. Report data quality plainly: which sensors you could not use and why.
6. Do not mention CSV, tables or methods. Write as a person who looked at the numbers.
7. EV charging has many stations: say how many you used, describe the typical
   pattern in general terms and name at most three stations.

Voice: first person singular, warm, plain, honest. No marketing tone, no
exclamation marks, no emojis. Use bold only for lot, station and sensor names.

Format: Markdown only, nothing before or after it.
  # Imidillo report, {month_label}
  two or three opening sentences
  ## Highlights          3 to 5 bullets
  ## Parking
  ## EV charging
  ## Air quality
  ## Data quality
  a one-line sign-off: Imidillo
Short paragraphs and bullets. 500 to 800 words. Language: {language}.
"""

FAMILY_NOTES = {
    "parking": "Parking occupancy in % of the lot's capacity (100 % = full).",
    "ev_charging": "EV charging point occupancy in % (each station has 2 points; 50 % = one point in use). "
                   "Values are hourly time-weighted shares.",
    "no2": "NO2 concentration in µg/m³.",
    "o3": "O3 concentration in µg/m³.",
    "pm10": "PM10 concentration in µg/m³.",
    "pm25": "PM2.5 concentration in µg/m³.",
}


# --------------------------------------------------------------------------- #
# Data export                                                                  #
# --------------------------------------------------------------------------- #

def _wide(series: dict, label: str) -> pd.DataFrame:
    cols = {s.name: s.values for s in series.values()}
    df = pd.DataFrame(cols)
    df.index = df.index.strftime("%Y-%m-%d %H:00")
    df.index.name = "local_hour"
    df.columns.name = label
    return df.round(1)


def export_data(year: int, month: int, out_dir) -> str:
    """Hourly series of both months as CSV blocks; also saved to out_dir."""
    py, pm = previous_of(year, month)
    windows = {"current": month_window(year, month), "previous": month_window(py, pm)}
    blocks = []
    for which, (start, end) in windows.items():
        label = month_label(year, month) if which == "current" else month_label(py, pm)
        fams = {"parking": parking.prepare_parking(start, end),
                "ev_charging": ev_charging.prepare_ev_charging(start, end)}
        air = air_quality.prepare_air_quality(start, end)
        fams.update({p: air[p] for p in ("no2", "o3", "pm10", "pm25") if air[p]})
        for key, series in fams.items():
            df = _wide(series, key)
            path = out_dir / f"data_{year}-{month:02d}_{which}_{key}.csv"
            df.to_csv(path)
            blocks.append(f"### {key} - {label} ({period_label(start, end)})\n{FAMILY_NOTES[key]}\n"
                          + df.to_csv(na_rep=""))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Conditions                                                                   #
# --------------------------------------------------------------------------- #

def write_raw(facts: dict, data_block: str) -> str:
    r = facts["report"]
    instr = RAW_INSTRUCTIONS.format(month_label=r["month_label"], previous_month_label=r["previous_month_label"],
                                    language=LANG.get(config.REPORT_LANGUAGE, "English"))
    return complete(instr, "DATA:\n" + data_block, model=config.WRITER_MODEL, label="raw-writer")


def judge(report_md: str, facts: dict, facts_json: str) -> dict:
    sections = {t: x for t, x in split_sections(report_md) if t in SCORED}
    out = {}
    for title in SCORED:
        text = sections.get(title)
        if not text:
            out[title] = {"missing": True}
            continue
        items = _verify_section(facts_json, title, text)
        body = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
        numbers = [n for n in re.findall(r"\d+(?:\.\d+)?", body) if n not in {"2025", "2026", "2027"}]
        out[title] = {
            "items": items,
            "n_claims": len(items),
            "supported": sum(i["verdict"] == "supported" for i in items),
            "unsupported": sum(i["verdict"] == "unsupported" for i in items),
            "unverifiable": sum(i["verdict"] == "unverifiable" for i in items),
            "numbers": len(numbers),
            "untraced_numbers": untraced_numbers(text, facts),
            "words": len(text.split()),
        }
    return out


def summarize(results: list[dict]) -> dict:
    agg = collections.defaultdict(lambda: collections.Counter())
    per_report = collections.defaultdict(list)
    for r in results:
        c = agg[r["condition"]]
        tot = collections.Counter()
        for sec in r["judgement"].values():
            if sec.get("missing"):
                c["missing_sections"] += 1
                continue
            for k in ("n_claims", "supported", "unsupported", "unverifiable", "numbers", "words"):
                tot[k] += sec[k]
            tot["untraced"] += len(sec["untraced_numbers"])
        c.update(tot)
        c["reports"] += 1
        per_report[r["condition"]].append(
            round(tot["unsupported"] / tot["n_claims"], 3) if tot["n_claims"] else None)
    out = {}
    for cond, c in agg.items():
        n = c["n_claims"] or 1
        out[cond] = {**dict(c),
                     "supported_rate": round(c["supported"] / n, 3),
                     "unsupported_rate": round(c["unsupported"] / n, 3),
                     "unverifiable_rate": round(c["unverifiable"] / n, 3),
                     "untraced_per_100_numbers": round(100 * c["untraced"] / (c["numbers"] or 1), 1),
                     "unsupported_rate_per_report": per_report[cond]}
    return out


def raw_rerun(run_dir, months: list[str]) -> int:
    """Re-write and re-judge only the raw condition inside an existing run,
    reusing that run's exported data blocks. Used after the staleness rule in
    RAW_INSTRUCTIONS was corrected to match the pipeline (pollutants only):
    the first raw reports had applied it to EV charging states as well.
    The corrected condition is stored as "raw_v2"; summary_final.json combines
    it with the run's pipeline conditions."""
    from pathlib import Path
    run_dir = Path(run_dir)
    config.LLM_TIMEOUT_S = max(config.LLM_TIMEOUT_S, 1200)
    ctx = {}
    for m in months:
        facts = load_facts(m)
        ctx[m] = {"facts": facts, "facts_json": json.dumps(slim_facts(facts), ensure_ascii=False, indent=1),
                  "block": (run_dir / f"raw_prompt_{m}.txt").read_text(encoding="utf-8")}
    jobs = [(m, k) for m in months for k in range(RUNS)]

    def _one(job):
        m, k = job
        c = ctx[m]
        text = _retry(write_raw, c["facts"], c["block"])
        (run_dir / f"{m}_raw_v2_{k}.md").write_text(text, encoding="utf-8")
        rep = {"month": m, "run": k, "condition": "raw_v2"}
        return {**rep, "judgement": _retry(judge, text, c["facts"], c["facts_json"])}

    v2_file = run_dir / "judgements_raw_v2.jsonl"
    if "--combine-only" in sys.argv:
        results = [json.loads(l) for l in v2_file.read_text(encoding="utf-8").splitlines()]
    else:
        print(f"[EXP-B] raw_v2: {len(jobs)} reports ...", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_one, jobs))
        with open(v2_file, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    main_file = run_dir / "judgements.jsonl"
    if not main_file.exists():
        print("[EXP-B] raw_v2 judged; the main run has not finished yet. Combine later with:\n"
              f"  python -m experiments.exp_b_raw_baseline --raw-rerun {run_dir} --combine-only", flush=True)
        return 0
    old = [json.loads(l) for l in main_file.read_text(encoding="utf-8").splitlines()]
    combined = [r for r in old if r["condition"] != "raw"] + results
    summary = {"by_condition": summarize(combined),
               "superseded": {"raw": summarize([r for r in old if r["condition"] == "raw"]).get("raw")},
               "note": "raw_v2 = raw with the staleness rule restricted to pollutants, as in the pipeline"}
    (run_dir / "summary_final.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def judge_only(run_dir) -> int:
    """Judge every saved report of a run, one result file per report, resumable.

    Written after the main run's judging stage hung on API timeouts at night:
    each report's judgement is saved as judge_<file>.json as soon as it is done,
    finished reports are skipped on restart, and a short timeout makes a hung
    call fail fast and retry. Produces summary_final.json over raw_v2,
    pipeline_draft and pipeline_final (raw, superseded, is summarised apart)."""
    from pathlib import Path
    run_dir = Path(run_dir)
    config.LLM_TIMEOUT_S = 300
    pat = re.compile(r"^(\d{4}-\d{2})_(raw|raw_v2|pipeline_draft|pipeline_final)_(\d+)\.md$")
    reports = []
    for f in sorted(run_dir.glob("*.md")):
        m = pat.match(f.name)
        if m and (m.group(2) != "raw" or "--include-superseded" in sys.argv):
            reports.append({"file": f, "month": m.group(1), "condition": m.group(2), "run": int(m.group(3))})
    # reuse judgements that already exist from the raw rerun
    v2_file = run_dir / "judgements_raw_v2.jsonl"
    if v2_file.exists():
        for r in (json.loads(l) for l in v2_file.read_text(encoding="utf-8").splitlines()):
            out = run_dir / f"judge_{r['month']}_raw_v2_{r['run']}.json"
            if not out.exists():
                out.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    facts_cache = {}

    def _one(rep):
        out = run_dir / f"judge_{rep['file'].stem}.json"
        if out.exists():
            return json.loads(out.read_text(encoding="utf-8"))
        m = rep["month"]
        if m not in facts_cache:
            f = load_facts(m)
            facts_cache[m] = (f, json.dumps(slim_facts(f), ensure_ascii=False, indent=1))
        facts, facts_json = facts_cache[m]
        res = {"month": m, "run": rep["run"], "condition": rep["condition"],
               "judgement": _retry(judge, rep["file"].read_text(encoding="utf-8"), facts, facts_json)}
        out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"[EXP-B] judged {rep['file'].name}", flush=True)
        return res

    todo = [r for r in reports if not (run_dir / f"judge_{r['file'].stem}.json").exists()]
    print(f"[EXP-B] {len(reports)} reports, {len(todo)} to judge", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_one, reports))
    current = [r for r in results if r["condition"] != "raw"]
    summary = {"by_condition": summarize(current),
               "superseded": {"raw": summarize([r for r in results if r["condition"] == "raw"]).get("raw")},
               "note": "raw_v2 = raw with the staleness rule restricted to pollutants, as in the pipeline"}
    (run_dir / "summary_final.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_condition"], indent=2))
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--judge-only":
        return judge_only(sys.argv[2])
    if len(sys.argv) > 2 and sys.argv[1] == "--raw-rerun":
        return raw_rerun(sys.argv[2], [a for a in sys.argv[3:] if not a.startswith("--")] or ["2026-08", "2026-09"])
    months = sys.argv[1:] or ["2026-08", "2026-09"]
    config.LLM_TIMEOUT_S = max(config.LLM_TIMEOUT_S, 1200)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = config.OUTPUT_DIR / "experiments" / "exp_b" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = {}
    for m in months:
        y, mo = parse_month(m)
        facts = load_facts(m)
        print(f"[EXP-B] exporting hourly data for {m} and the month before ...", flush=True)
        block = export_data(y, mo, out_dir)
        (out_dir / f"raw_prompt_{m}.txt").write_text(block, encoding="utf-8")
        ctx[m] = {"facts": facts, "facts_json": json.dumps(slim_facts(facts), ensure_ascii=False, indent=1),
                  "block": block}
        print(f"[EXP-B] {m}: raw data block {len(block) / 1000:.0f}k characters", flush=True)
    print(f"[EXP-B] writing {len(months) * RUNS * 2} reports ...", flush=True)

    # 1. write: raw and pipeline drafts
    jobs = [(m, k, cond) for m in months for k in range(RUNS) for cond in ("raw", "pipeline_draft")]

    def _write(job):
        m, k, cond = job
        c = ctx[m]
        text = _retry(write_raw, c["facts"], c["block"]) if cond == "raw" else _retry(write_report, c["facts"])
        (out_dir / f"{m}_{cond}_{k}.md").write_text(text, encoding="utf-8")
        return {"month": m, "run": k, "condition": cond, "text": text}

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(_write, jobs))

    # 2. pipeline_final: production verify-and-fix on every draft
    def _final(rep):
        c = ctx[rep["month"]]
        text, history = _retry(verify_and_fix, c["facts"], rep["text"])
        (out_dir / f"{rep['month']}_pipeline_final_{rep['run']}.md").write_text(text, encoding="utf-8")
        return {"month": rep["month"], "run": rep["run"], "condition": "pipeline_final", "text": text,
                "rounds": len(history), "cut": len(history[-1].stripped)}

    print("[EXP-B] verify-and-fix on the pipeline drafts ...", flush=True)
    drafts = [r for r in reports if r["condition"] == "pipeline_draft"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports += list(pool.map(_final, drafts))

    print(f"[EXP-B] judging {len(reports)} reports ...", flush=True)
    # 3. judge every report against the true facts
    def _judge(rep):
        c = ctx[rep["month"]]
        return {**{k: v for k, v in rep.items() if k != "text"}, "judgement": _retry(judge, rep["text"], c["facts"], c["facts_json"])}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_judge, reports))
    with open(out_dir / "judgements.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {"by_condition": summarize(results),
               "config": {"months": months, "runs": RUNS, "scored_sections": SCORED, "model": config.WRITER_MODEL,
                          "reasoning": config.LLM_REASONING, "run": run_id}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
