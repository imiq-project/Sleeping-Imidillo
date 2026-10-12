"""
Experiment A: does the verifier catch fabricated claims?

Protocol
  1. For each month, generate claims from that month's facts.json with
     experiments/claims.py: PER_TYPE claims for each of six error types, plus
     N_CONTROLS true claims. Seeded, so the set is reproducible.
  2. Insert ONE claim into the matching section of that month's APPROVED
     report (report.md), as an extra paragraph at the end of the section.
  3. Verify that section exactly as production does (same prompt, model,
     reasoning effort, full slim facts): report.verifier._verify_section.
  4. Find the verifier's item for the inserted sentence (best string match)
     and record its verdict. Also record whether the deterministic number
     tracer (untraced_numbers) alone would have flagged the sentence.
  One claim per call keeps trials independent.

Outcomes per error claim:
  caught      verdict "unsupported" (this is what triggers a rewrite or a cut)
  flagged     "unsupported" or "unverifiable"
  missed      "supported", or the verifier did not list the sentence at all
Outcomes per control claim:
  false alarm verdict "unsupported"

Run:
    python -m experiments.exp_a_injection 2026-08 2026-09
Writes output/experiments/exp_a/<run>/trials.jsonl and summary.json.
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config
from experiments.claims import ERROR_TYPES, TEMPLATE_VERSION, ClaimGenerator, as_dicts
from report.facts import load_facts, slim_facts
from report.verifier import _verify_section, split_sections, untraced_numbers

SEED = 20261007
PER_TYPE = 8
N_CONTROLS = 16
WORKERS = 8


def _norm(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"^\s*[-*]\s+", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _match(sentence: str, items: list[dict]) -> tuple[dict | None, float]:
    target = _norm(sentence)
    best, best_r = None, 0.0
    for it in items:
        r = difflib.SequenceMatcher(None, target, _norm(it["claim"])).ratio()
        if _norm(it["claim"]) in target or target in _norm(it["claim"]):
            r = max(r, 0.95)
        if r > best_r:
            best, best_r = it, r
    return (best, best_r) if best_r >= 0.6 else (None, best_r)


def _inject(report_md: str, section: str, sentence: str) -> str | None:
    for title, text in split_sections(report_md):
        if title == section:
            return text.rstrip() + "\n\n" + sentence
    return None


def run_trial(claim: dict, report_md: str, facts: dict, facts_json: str) -> dict:
    text = _inject(report_md, claim["section"], claim["sentence"])
    out = dict(claim)
    out["tracer_flags"] = untraced_numbers(claim["sentence"], facts)
    if text is None:
        out.update(status="no_section", verdict=None)
        return out
    t0 = time.perf_counter()
    try:
        items = _verify_section(facts_json, claim["section"], text)
    except Exception as e:  # noqa: BLE001
        out.update(status="error", error=f"{type(e).__name__}: {e}", verdict=None)
        return out
    item, ratio = _match(claim["sentence"], items)
    out.update(status="ok", seconds=round(time.perf_counter() - t0, 1), match_ratio=round(ratio, 2),
               verdict=item["verdict"] if item else "not_listed",
               reason=item["reason"] if item else None,
               evidence=item["evidence"] if item else None,
               section_items=len(items),
               other_unsupported=sum(1 for it in items if it is not item and it["verdict"] == "unsupported"))
    return out


def summarize(trials: list[dict]) -> dict:
    ok = [t for t in trials if t["status"] == "ok"]
    by = collections.defaultdict(list)
    for t in ok:
        by[t["error_type"]].append(t)
    rows = {}
    for etype in ERROR_TYPES:
        ts = by.get(etype, [])
        n = len(ts)
        caught = sum(t["verdict"] == "unsupported" for t in ts)
        flagged = sum(t["verdict"] in ("unsupported", "unverifiable") for t in ts)
        tracer = sum(bool(t["tracer_flags"]) for t in ts)
        either = sum(t["verdict"] == "unsupported" or bool(t["tracer_flags"]) for t in ts)
        rows[etype] = {"n": n, "caught": caught, "flagged": flagged, "tracer_only": tracer,
                       "caught_or_tracer": either,
                       "caught_rate": round(caught / n, 3) if n else None,
                       "flagged_rate": round(flagged / n, 3) if n else None,
                       "tracer_rate": round(tracer / n, 3) if n else None}
    errs = [t for t in ok if t["is_error"]]
    ctrl = by.get("none", [])
    total = {
        "errors_n": len(errs),
        "errors_caught": sum(t["verdict"] == "unsupported" for t in errs),
        "errors_flagged": sum(t["verdict"] in ("unsupported", "unverifiable") for t in errs),
        "errors_tracer": sum(bool(t["tracer_flags"]) for t in errs),
        "controls_n": len(ctrl),
        "controls_supported": sum(t["verdict"] == "supported" for t in ctrl),
        "controls_unverifiable": sum(t["verdict"] == "unverifiable" for t in ctrl),
        "controls_false_alarm": sum(t["verdict"] == "unsupported" for t in ctrl),
        "not_listed": sum(t["verdict"] == "not_listed" for t in ok),
        "failed_calls": sum(t["status"] != "ok" for t in trials),
        "median_seconds": sorted(t["seconds"] for t in ok)[len(ok) // 2] if ok else None,
    }
    for k in ("errors_caught", "errors_flagged", "errors_tracer"):
        total[k + "_rate"] = round(total[k] / total["errors_n"], 3) if total["errors_n"] else None
    total["controls_false_alarm_rate"] = (round(total["controls_false_alarm"] / total["controls_n"], 3)
                                          if total["controls_n"] else None)
    return {"by_error_type": rows, "total": total}


def main() -> int:
    months = sys.argv[1:] or ["2026-08", "2026-09"]
    config.LLM_TIMEOUT_S = max(config.LLM_TIMEOUT_S, 900)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = config.OUTPUT_DIR / "experiments" / "exp_a" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for month in months:
        facts = load_facts(month)
        report_md = (config.OUTPUT_DIR / month / "report.md").read_text(encoding="utf-8")
        facts_json = json.dumps(slim_facts(facts), ensure_ascii=False, indent=1)
        claims = as_dicts(ClaimGenerator(facts, SEED).build(PER_TYPE, N_CONTROLS))
        (out_dir / f"claims_{month}.json").write_text(json.dumps(claims, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
        jobs += [(c, report_md, facts, facts_json) for c in claims]
    print(f"[EXP-A] {len(jobs)} trials over {months}, model {config.VERIFIER_MODEL}, "
          f"reasoning {config.LLM_REASONING}, {WORKERS} in parallel -> {out_dir}")

    trials = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, \
            open(out_dir / "trials.jsonl", "w", encoding="utf-8") as fh:
        futures = [pool.submit(run_trial, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            t = fut.result()
            trials.append(t)
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
            fh.flush()
            mark = "ERR" if t["is_error"] else "CTL"
            print(f"[EXP-A] {i:3d}/{len(jobs)} {mark} {t['error_type']:12s} -> {t.get('verdict')}", flush=True)

    summary = summarize(trials)
    summary["config"] = {"months": months, "seed": SEED, "per_type": PER_TYPE, "n_controls": N_CONTROLS,
                         "model": config.VERIFIER_MODEL, "reasoning": config.LLM_REASONING, "run": run_id,
                         "templates": TEMPLATE_VERSION}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
