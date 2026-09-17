
# Verifier LLM: every claim in the draft is checked against the facts.

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import config
from report.facts import load_facts, slim_facts
from report.llm import complete
from report.writer import INSTRUCTIONS as WRITER_INSTRUCTIONS, LANGUAGES

MAX_ROUNDS = 2   # revision rounds before unsupported sentences are cut

INSTRUCTIONS = """\
You are the verifier for Imidillo's monthly city report. You receive FACTS
(JSON, the only source of truth) and a DRAFT (Markdown written from FACTS).
Check every checkable sentence of the DRAFT against FACTS. Bullets count as
sentences.

For each sentence that states something checkable (a number, an hour, a
name, a ranking, a direction of change, a significance judgement, an
exclusion reason, a cause) return one item:
  claim     the sentence, quoted exactly as in the draft (without the leading
            "- " and without Markdown bold markers)
  verdict   "supported"     every number, hour, name and direction matches
                            FACTS, and any increase/decrease/fuller/emptier/
                            rise/drop wording corresponds to significant: true
            "unsupported"   a number, hour, name or direction contradicts
                            FACTS; or the sentence asserts a change or effect
                            FACTS marks significant: false; or it gives a
                            cause or explanation; or it uses a number FACTS
                            does not contain
            "unverifiable"  factual-sounding, but FACTS can neither confirm
                            nor deny it
  evidence  the FACTS path(s) checked, e.g. parking_patterns.lots.Ulrichshaus.peak_hour
  reason    one short sentence; for unsupported, say what FACTS actually says

Skip sentences with nothing checkable: headings, greetings, the sign-off,
"the data cannot tell me why".

Conventions that ARE supported: "13.7 points emptier" for -13.7; "no
measurable change" for significant: false; "36%" for 36.0; hours as 11:00
for peak_hour 11; a lot's "name" value instead of its key. Reasons that FACTS
itself states (capacity_note, exclusion reasons in data_quality, report.partial
and report.period_label) are supported, not "causes". Do not invent
leniency beyond this: when in doubt, unsupported.
Return JSON matching the schema, nothing else.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["supported", "unsupported", "unverifiable"]},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "verdict", "evidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Layer 1: deterministic number tracing                                        #
# --------------------------------------------------------------------------- #

def untraced_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in the text that appear nowhere in the facts. Signs are ignored
    ('1.1 points emptier' for -1.1); clock times, '95%' and years are skipped."""
    vals: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            a = abs(o)
            vals.update({f"{a}", f"{a:.1f}", f"{a:.0f}"})
        elif isinstance(o, str):
            vals.update(re.findall(r"\d+(?:\.\d+)?", o))

    walk(facts)
    vals.update({"95", "2025", "2026", "2027"})
    body = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    return sorted({n for n in re.findall(r"\d+(?:\.\d+)?", body) if n not in vals}, key=float)


# --------------------------------------------------------------------------- #
# Layer 2: the verifier model                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Verification:
    items: list[dict]
    untraced: list[str]
    round: int = 0
    stripped: list[str] = field(default_factory=list)

    def count(self, verdict: str) -> int:
        return sum(1 for i in self.items if i["verdict"] == verdict)

    @property
    def unsupported(self) -> list[dict]:
        return [i for i in self.items if i["verdict"] == "unsupported"]

    @property
    def passed(self) -> bool:
        return not self.unsupported and not self.untraced

    def summary(self) -> str:
        return (f"round {self.round}: {self.count('supported')} supported, "
                f"{self.count('unsupported')} unsupported, {self.count('unverifiable')} unverifiable, "
                f"{len(self.untraced)} untraced numbers"
                + (f", {len(self.stripped)} sentences cut" if self.stripped else ""))

    def to_dict(self) -> dict:
        return {"round": self.round, "passed": self.passed, "untraced_numbers": self.untraced,
                "stripped": self.stripped, "items": self.items}


def verify(facts: dict, draft: str, round_no: int = 0) -> Verification:
    user_input = ("FACTS:\n" + json.dumps(slim_facts(facts), ensure_ascii=False, indent=1)
                  + "\n\nDRAFT:\n" + draft)
    raw = complete(INSTRUCTIONS, user_input, model=config.VERIFIER_MODEL,
                   label="verifier", json_schema=SCHEMA, schema_name="verification")
    items = json.loads(raw)["items"]
    return Verification(items=items, untraced=untraced_numbers(draft, facts), round=round_no)


# --------------------------------------------------------------------------- #
# Fixing                                                                       #
# --------------------------------------------------------------------------- #

def revise(facts: dict, draft: str, v: Verification) -> str:
    """Send the flagged sentences back to the writer; everything else stays."""
    r = facts["report"]
    instructions = WRITER_INSTRUCTIONS.format(
        month_label=r["month_label"],
        language=LANGUAGES.get(config.REPORT_LANGUAGE, config.REPORT_LANGUAGE))
    problems = "\n".join(f'- "{i["claim"]}"\n  problem: {i["reason"]}' for i in v.unsupported)
    if v.untraced:
        problems += f"\n- numbers that do not exist in FACTS: {', '.join(v.untraced)}"
    instructions += (
        "\n\nREVISION. A verifier checked your previous draft against FACTS and flagged "
        "the sentences below. Fix each one so it is supported by FACTS, or delete it. "
        "Keep every other sentence exactly as it was. Output the complete Markdown.\n"
        + problems
    )
    return complete(instructions, "FACTS:\n" + json.dumps(slim_facts(facts), ensure_ascii=False, indent=1)
                    + "\n\nPREVIOUS DRAFT:\n" + draft, model=config.WRITER_MODEL, label="writer-revise")


def strip_unsupported(text: str, v: Verification) -> tuple[str, list[str]]:
    """Last resort: cut the unsupported sentences out of the text."""
    removed = []
    for item in v.unsupported:
        claim = item["claim"].strip()
        plain = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # compare without bold markers
        if claim and claim in plain:
            # remove from the real text by locating the bold-stripped position
            idx = plain.find(claim)
            # rebuild: strip bold everywhere is acceptable for the final text
            text = plain[:idx] + plain[idx + len(claim):]
            removed.append(claim)
    text = re.sub(r"^- *\n", "", text, flags=re.MULTILINE)   # empty bullets left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, removed


def verify_and_fix(facts: dict, draft: str) -> tuple[str, list[Verification]]:
    """Verify; revise while needed (MAX_ROUNDS); cut what still fails.
    Returns (approved text, verification history)."""
    text, history = draft, []
    for round_no in range(MAX_ROUNDS + 1):
        v = verify(facts, text, round_no)
        history.append(v)
        print(f"[verifier] {v.summary()}")
        if v.passed:
            return text, history
        if round_no < MAX_ROUNDS:
            text = revise(facts, text, v)
    text, removed = strip_unsupported(text, history[-1])
    history[-1].stripped = removed
    print(f"[verifier] cut {len(removed)} unsupported sentence(s) after {MAX_ROUNDS} revisions")
    return text, history


# --------------------------------------------------------------------------- #
# Manual check                                                                 #
# --------------------------------------------------------------------------- #

INJECTED = [
    "Tiefgarage Ulrichshaus was busiest at 15:00 with 81.3% occupancy.",
    "Occupancy at Elbauenpark rose because of the summer fair.",
]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m report.verifier <YYYY-MM> [--inject]")
        return 1
    month = sys.argv[1]
    facts = load_facts(month)
    draft_path = config.OUTPUT_DIR / month / "draft.md"
    draft = draft_path.read_text(encoding="utf-8")
    if "--inject" in sys.argv:
        draft = draft.replace("## Data quality", "\n".join(INJECTED) + "\n\n## Data quality")
        print(f"[verifier] injected {len(INJECTED)} fabricated sentences into the draft")

    text, history = verify_and_fix(facts, draft)

    for v in history:
        print(f"\n--- {v.summary()} ---")
        for i in v.items:
            if i["verdict"] != "supported":
                print(f"  [{i['verdict'].upper():12s}] {i['claim'][:90]}")
                print(f"  {'':14s} {i['reason']}  ({i['evidence']})")
        if v.untraced:
            print(f"  untraced numbers: {v.untraced}")

    out_dir = config.OUTPUT_DIR / month
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    (out_dir / "verification.json").write_text(
        json.dumps([v.to_dict() for v in history], indent=2, ensure_ascii=False), encoding="utf-8")
    final = history[-1]
    print(f"\n[verifier] final: {'APPROVED' if final.passed or final.stripped else 'NOT APPROVED'}; "
          f"report.md and verification.json written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())