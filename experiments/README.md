# Experiments

Evaluation of the report pipeline, written for the paper
*From Sensor Streams to Trustworthy City Reports* (IMIQ "Places of Tomorrow",
24 Nov 2026) and meant to be extended for a larger venue.

Both experiments read the saved `output/<month>/facts.json` and `report.md` of
the evaluated months (produced by `python -m jobs.monthly_report <month> --dry-run`),
use the production model and reasoning effort from `.env`, and write every
prompt, output and judgement to `output/experiments/<exp>/<run>/`.

## A. Fabrication injection (`exp_a_injection.py`)

Question: how reliably does the verifier reject a false claim, and how often
does it reject a true one?

- Claims are generated from the facts by `claims.py`: 8 per error type and 16
  true controls per month, seeded (20261007). Error types: wrong number, wrong
  hour, non-significant change stated as a change, reversed direction,
  invented cause, value attributed to the wrong sensor.
- Each claim is appended to its section of the approved report, and that
  section is verified exactly as in production. One claim per call.
- Outcome: the verifier's verdict on the inserted sentence. "Caught" means
  "unsupported", which triggers a rewrite or a cut in production. The
  deterministic number tracer is scored on the same sentences.

## B. Raw-data baseline (`exp_b_raw_baseline.py`)

Question: what does computing the statistics outside the LLM buy?

- `raw`: the writer gets the cleaned hourly series of the report month and the
  previous month (the same series the statistics layer uses) and must compute
  everything itself. The task rules, health limits and data-quality rules are
  the same as in the pipeline.
- `pipeline_draft`: the production writer, fed only facts.json.
- `pipeline_final`: that draft after the production verify-and-fix loop.
- 3 reports per condition and month. The Parking, EV charging and Air quality
  sections are scored against the true facts by the verifier as a judge.
  Experiment A gives the judge's error rates.

## Run

```bash
python -m experiments.exp_a_injection 2026-08 2026-09
python -m experiments.exp_b_raw_baseline 2026-08 2026-09
```
