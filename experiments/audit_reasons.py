"""
Was a fabrication caught for the RIGHT reason?

A verdict of "unsupported" counts as a catch, but the verifier could reject a
sentence for an unrelated reason (e.g. an ambiguous wording) and still be
"right". This script checks, per error trial, whether the verifier's reason
mentions the planted error, using per-type evidence:
  number        the true value from `detail` appears in the reason
  hour          the true peak hour appears in the reason (as HH or HH:00)
  significance  the reason says the change is not significant / not measurable,
                or quotes the confidence interval
  direction     the reason names the true direction or sign
  cause         the reason objects to a cause / explanation
  entity        the reason mentions the true value or the other sensor
Trials without automatic evidence are listed for manual review; their manual
labels go into manual_labels.json ({trial_id: true|false}) next to trials.jsonl.

Run:
    python -m experiments.audit_reasons output/experiments/exp_a/<run>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _auto(t: dict) -> bool | None:
    reason = (t.get("reason") or "").lower()
    d = t["detail"]
    et = t["error_type"]
    if et == "number":
        true = re.search(r"([\d.]+) -> ", d)
        if true and true.group(1).rstrip("0").rstrip(".") in reason.replace(".0 ", " "):
            return True
        return true is not None and true.group(1) in reason or None
    if et == "hour":
        h = int(re.search(r"peak hour (\d+)", d).group(1))
        return bool(re.search(rf"\b0?{h}(:00)?\b", reason)) or None
    if et == "significance":
        return bool(re.search(r"not significant|significant: false|significant is false|significant false|"
                              r"significance is false|no measurable|not measurable|"
                              r"ci95|confidence interval|includes 0|includes zero|insignificant", reason)) or None
    if et == "direction":
        return bool(re.search(r"opposite|lower|higher|decrease|increase|less|more|negative|positive|"
                              r"emptier|fuller|busier|cleaner|faster|slower", reason)) or None
    if et == "cause":
        return bool(re.search(r"cause|explanation|reason|why|attribut|because", reason)) or None
    if et == "entity":
        nums = re.findall(r"\(([\d.]+)\)", d)
        other = re.search(r"of (\S+) ", d)
        hit = any(n in reason for n in nums) or (other and other.group(1).lower()[:8] in reason.replace(" ", ""))
        # "X peaks at 11:00 with 72.5%, not 13:00 with 36.0%": the reason states the
        # named sensor's true value and rejects the claimed one -- the right objection
        claimed = re.findall(r"\d+\.\d", t["sentence"])
        stated = re.findall(r"\d+\.\d", reason)
        hit = hit or (bool(re.search(r"\bnot\b|rather than|instead|does not match|differs", reason))
                      and any(n not in claimed for n in stated))
        return bool(hit) or None
    return None


def main() -> int:
    run = Path(sys.argv[1])
    trials = [json.loads(l) for l in (run / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    manual_path = run / "manual_labels.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    rows, review = [], []
    for t in trials:
        if not t["is_error"] or t.get("verdict") != "unsupported":
            continue
        auto = _auto(t)
        label = manual.get(t["id"], auto)
        rows.append({"id": t["id"], "type": t["error_type"], "auto": auto, "right_reason": label})
        if label is None:
            review.append(t)
    by = {}
    for r in rows:
        b = by.setdefault(r["type"], {"caught": 0, "right_reason": 0, "unresolved": 0})
        b["caught"] += 1
        b["right_reason"] += r["right_reason"] is True
        b["unresolved"] += r["right_reason"] is None
    (run / "reason_audit.json").write_text(json.dumps({"by_type": by, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(by, indent=2))
    for t in review:
        print(f"\n[REVIEW {t['id']}] {t['error_type']} | {t['detail']}\n  sentence: {t['sentence']}\n  reason:   {t['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
