"""
Claim generator for the verifier experiments.

Builds report-style sentences directly from a month's facts.json, either TRUE
(a control the verifier should accept) or with ONE deliberate error of a known
type (a fabrication it should reject). Everything is template-based and
seeded, so the claim set is reproducible and every label is known by
construction; no LLM is involved in making the test items.

Error taxonomy (one error per claim):
  number        a value (mean, peak level, exceedance count) changed by 20-50 %
  hour          the busiest hour moved by 4-9 hours
  significance  a non-significant month-on-month change stated as a real change,
                with the TRUE magnitude (the number itself is in the facts)
  direction     a significant change stated in the opposite direction, true magnitude
  cause         a true, significant change plus an invented explanation
  entity        a true statement about one sensor attributed to another sensor
                of the same family
The "number" and "entity" errors may or may not introduce a number that is
absent from the facts; that is recorded per claim (`number_in_facts`) so the
deterministic number tracer and the LLM verifier can be compared fairly.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

SECTION_OF_FAMILY = {
    "parking": "Parking",
    "ev_charging": "EV charging",
    "traffic": "Traffic",
    "air_no2": "Air quality",
    "air_o3": "Air quality",
    "air_pm10": "Air quality",
    "air_pm25": "Air quality",
}

CAUSES = {
    "parking": ["the start of the autumn trade fair season", "road works on the Magdeburger Ring",
                "a change in parking fees", "the school holidays ending"],
    "ev_charging": ["new electric car-sharing vehicles in the area", "a price change by the operator",
                    "more commuters switching to electric cars", "a faulty charger at a nearby station"],
    "traffic": ["construction on the Strombrücke", "the end of the summer holidays",
                "a new traffic light schedule", "diverted traffic from the B1"],
    "air_no2": ["heavier commuter traffic", "a stable high-pressure weather pattern",
                "construction machinery nearby", "lower wind speeds"],
    "air_o3": ["more sunshine and higher temperatures", "a heat wave", "lower nitrogen oxide levels",
               "regional ozone transport from the south"],
}
ERROR_TYPES = ["number", "hour", "significance", "direction", "cause", "entity"]
TEMPLATE_VERSION = "v2"   # v1 (first run, 2026-10-07): shift/no-shift sentences did not name the variable


@dataclass
class Claim:
    id: str
    month: str
    family: str
    series: str
    section: str
    error_type: str            # "none" for controls
    is_error: bool
    sentence: str
    detail: str


# --------------------------------------------------------------------------- #
# Formatting helpers                                                           #
# --------------------------------------------------------------------------- #

def _v(x: float, unit: str) -> str:
    return f"{x:.1f}%" if unit == "%" else f"{x:.1f} {unit}"


def _d(x: float, delta_unit: str) -> str:
    return f"{abs(x):.1f} {delta_unit}"


def _var(fam: dict) -> str:
    return fam["variable"]


def _peak_sentence(name: str, fam: dict, hour: int, value: float) -> str:
    return (f"On a typical weekday, {name} reached its highest {_var(fam)} at {hour:02d}:00, "
            f"at {_v(value, fam['unit'])}.")


def _mean_sentence(name: str, fam: dict, value: float) -> str:
    return f"Over this period, the {_var(fam)} at {name} averaged {_v(value, fam['unit'])}."


def _pollutant(fam: dict) -> str:
    return fam["label"].split(": ")[-1]          # "Air quality: NO2" -> "NO2"


def _shift_sentence(name: str, fam: dict, prev_label: str, delta: float, up: bool) -> str:
    # v2 (2026-10-07): always name the variable. Station names repeat across
    # families ("Schleinufer" is a traffic sensor and an air station, which
    # measures several pollutants), and an unnamed variable made true
    # control sentences ambiguous in the first run.
    word = "higher" if up else "lower"
    return (f"Compared with {prev_label}, the {_var(fam)} at {name} was {word} by "
            f"{_d(delta, fam['delta_unit'])} at equal hour and weekday.")


def _noshift_sentence(name: str, fam: dict, prev_label: str) -> str:
    return (f"Compared with {prev_label}, the {_var(fam)} at {name} showed no measurable change "
            f"at equal hour and weekday.")


def _threshold_sentence(station: str, fam: dict, t: dict) -> str:
    return (f"At {station}, the {t['limit']} for {_pollutant(fam)} of {t['value']} {fam['unit']} "
            f"was exceeded on {t['count_over']} {t['count_unit']}.")


# --------------------------------------------------------------------------- #
# Generator                                                                    #
# --------------------------------------------------------------------------- #

class ClaimGenerator:
    def __init__(self, facts: dict, seed: int):
        self.facts = facts
        self.month = facts["report"]["month"]
        self.prev_label = facts["report"]["previous_month_label"]
        self.rng = random.Random(f"{seed}-{self.month}")
        self.families = {k: f for k, f in facts["families"].items()
                         if k in SECTION_OF_FAMILY and f["patterns"]["series"]}
        self._n = 0

    def _claim(self, family: str, series: str, etype: str, sentence: str, detail: str) -> Claim:
        self._n += 1
        return Claim(id=f"{self.month}-{self._n:03d}", month=self.month, family=family, series=series,
                     section=SECTION_OF_FAMILY[family], error_type=etype,
                     is_error=etype != "none", sentence=sentence, detail=detail)

    def _pick_family(self, need) -> tuple[str, dict] | None:
        options = [(k, f) for k, f in self.families.items() if need(f)]
        return self.rng.choice(options) if options else None

    def _series(self, fam: dict) -> tuple[str, dict]:
        return self.rng.choice(sorted(fam["patterns"]["series"].items()))

    def _cmp(self, fam: dict, significant: bool) -> list[tuple[str, dict]]:
        return [(k, c) for k, c in sorted(fam["comparison"]["series"].items())
                if c["adjusted_shift"]["significant"] == significant
                and abs(c["adjusted_shift"]["delta"]) >= 0.1]

    # ---- controls ----------------------------------------------------------- #

    def control(self) -> Claim:
        kind = self.rng.choice(["mean", "peak", "shift", "noshift", "threshold"])
        if kind == "shift":
            pick = self._pick_family(lambda f: self._cmp(f, True))
            if pick:
                fk, fam = pick
                k, c = self.rng.choice(self._cmp(fam, True))
                d = c["adjusted_shift"]["delta"]
                return self._claim(fk, k, "none", _shift_sentence(c["name"], fam, self.prev_label, d, d > 0),
                                   f"true significant shift {d}")
        if kind == "noshift":
            pick = self._pick_family(lambda f: self._cmp(f, False))
            if pick:
                fk, fam = pick
                k, c = self.rng.choice(self._cmp(fam, False))
                return self._claim(fk, k, "none", _noshift_sentence(c["name"], fam, self.prev_label),
                                   "true non-significant shift")
        if kind == "threshold":
            pick = self._pick_family(lambda f: any(f.get("thresholds", {}).values()))
            if pick:
                fk, fam = pick
                st = self.rng.choice(sorted(k for k, v in fam["thresholds"].items() if v))
                t = self.rng.choice(fam["thresholds"][st])
                name = fam["patterns"]["series"].get(st, {}).get("name", st)
                return self._claim(fk, st, "none", _threshold_sentence(name, fam, t), "true threshold")
        fk, fam = self.rng.choice(sorted(self.families.items()))
        k, s = self._series(fam)
        if kind == "peak":
            return self._claim(fk, k, "none", _peak_sentence(s["name"], fam, s["peak_hour"], s["peak_value"]),
                               "true peak")
        return self._claim(fk, k, "none", _mean_sentence(s["name"], fam, s["mean"]), "true mean")

    # ---- errors ------------------------------------------------------------- #

    def error(self, etype: str) -> Claim | None:
        return getattr(self, f"_err_{etype}")()

    def _err_number(self) -> Claim:
        fk, fam = self.rng.choice(sorted(self.families.items()))
        k, s = self._series(fam)
        factor = self.rng.choice([self.rng.uniform(1.2, 1.5), self.rng.uniform(0.5, 0.8)])
        if self.rng.random() < 0.5:
            true, cap = s["mean"], 100.0 if fam["unit"] == "%" else 1e9
            wrong = round(min(cap, true * factor), 1)
            if abs(wrong - true) < 0.5:
                wrong = round(true + 5.0, 1)
            return self._claim(fk, k, "number", _mean_sentence(s["name"], fam, wrong), f"mean {true} -> {wrong}")
        true, cap = s["peak_value"], 100.0 if fam["unit"] == "%" else 1e9
        wrong = round(min(cap, true * factor), 1)
        if abs(wrong - true) < 0.5:
            wrong = round(true + 5.0, 1)
        return self._claim(fk, k, "number", _peak_sentence(s["name"], fam, s["peak_hour"], wrong),
                           f"peak value {true} -> {wrong}")

    def _err_hour(self) -> Claim:
        fk, fam = self.rng.choice(sorted(self.families.items()))
        k, s = self._series(fam)
        shift = self.rng.choice([4, 5, 6, 7, 8, 9]) * self.rng.choice([-1, 1])
        hour = (s["peak_hour"] + shift) % 24
        return self._claim(fk, k, "hour", _peak_sentence(s["name"], fam, hour, s["peak_value"]),
                           f"peak hour {s['peak_hour']} -> {hour}")

    def _err_significance(self) -> Claim | None:
        pick = self._pick_family(lambda f: self._cmp(f, False))
        if not pick:
            return None
        fk, fam = pick
        k, c = self.rng.choice(self._cmp(fam, False))
        d = c["adjusted_shift"]["delta"]
        return self._claim(fk, k, "significance", _shift_sentence(c["name"], fam, self.prev_label, d, d > 0),
                           f"non-significant shift {d} (CI {c['adjusted_shift']['ci95']}) stated as change")

    def _err_direction(self) -> Claim | None:
        pick = self._pick_family(lambda f: self._cmp(f, True))
        if not pick:
            return None
        fk, fam = pick
        k, c = self.rng.choice(self._cmp(fam, True))
        d = c["adjusted_shift"]["delta"]
        return self._claim(fk, k, "direction", _shift_sentence(c["name"], fam, self.prev_label, d, d < 0),
                           f"significant shift {d} stated in the opposite direction")

    def _err_cause(self) -> Claim:
        pick = self._pick_family(lambda f: self._cmp(f, True))
        if pick:
            fk, fam = pick
            k, c = self.rng.choice(self._cmp(fam, True))
            d = c["adjusted_shift"]["delta"]
            base = _shift_sentence(c["name"], fam, self.prev_label, d, d > 0)
        else:
            fk, fam = self.rng.choice(sorted(self.families.items()))
            k, s = self._series(fam)
            base = _peak_sentence(s["name"], fam, s["peak_hour"], s["peak_value"])
        cause = self.rng.choice(CAUSES.get(fk, CAUSES["parking"]))
        return self._claim(fk, k, "cause", base[:-1] + f", which was caused by {cause}.", f"invented cause: {cause}")

    def _err_entity(self) -> Claim | None:
        fams = [(fk, f) for fk, f in sorted(self.families.items()) if len(f["patterns"]["series"]) >= 2]
        if not fams:
            return None
        fk, fam = self.rng.choice(fams)
        items = sorted(fam["patterns"]["series"].items())
        for _ in range(20):
            (ka, a), (kb, b) = self.rng.sample(items, 2)
            if abs(a["mean"] - b["mean"]) >= 2.0 and a["peak_hour"] != b["peak_hour"]:
                if self.rng.random() < 0.5:
                    sent = _mean_sentence(b["name"], fam, a["mean"])
                    detail = f"mean of {ka} ({a['mean']}) attributed to {kb} ({b['mean']})"
                else:
                    sent = _peak_sentence(b["name"], fam, a["peak_hour"], a["peak_value"])
                    detail = f"peak of {ka} attributed to {kb}"
                return self._claim(fk, kb, "entity", sent, detail)
        return None

    # ---- batch -------------------------------------------------------------- #

    def build(self, per_error_type: int, n_controls: int) -> list[Claim]:
        claims: list[Claim] = []
        for etype in ERROR_TYPES:
            made, tries = 0, 0
            while made < per_error_type and tries < per_error_type * 10:
                tries += 1
                c = self.error(etype)
                if c and c.sentence not in {x.sentence for x in claims}:
                    claims.append(c)
                    made += 1
        made, tries = 0, 0
        while made < n_controls and tries < n_controls * 10:
            tries += 1
            c = self.control()
            if c.sentence not in {x.sentence for x in claims}:
                claims.append(c)
                made += 1
        return claims


def as_dicts(claims: list[Claim]) -> list[dict]:
    return [asdict(c) for c in claims]
