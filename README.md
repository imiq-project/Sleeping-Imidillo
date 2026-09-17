# Sleeping Imidillo

Imidillo is the Magdeburg city assistant of the IMIQ project. This repository is
its "other job": once a month, unattended, it emails the project team a short
city report built from Magdeburg's live sensor network. A systemd timer wakes
one Python script, the script does its work and exits. No server, no LangGraph;
this project is fully separate from the chatbot repo.

The report is honest by construction: the statistics are computed
deterministically, an LLM turns the numbers into prose, and a second LLM checks
every sentence against those numbers before anything is sent. If the data cannot
support a claim, the claim is not in the report.

## How it works

```
systemd timer (last Monday of the month, 10:00)
  └─ python -m jobs.monthly_report
       1. report/facts.py      QuantumLeap + Orion -> data/cleaning.py -> Statistical_models/*
                               -> output/<month>/facts.json   (every number with CI, p-value, coverage)
       2. report/writer.py     LLM writes the report; facts.json is its ONLY input
       3. report/verifier.py   LLM checks each sentence: supported / unsupported / unverifiable;
                               unsupported ones go back to the writer (2 rounds) or are cut
       4. report/render.py     HTML email (inline styles, CID images), charts, PDF
       5. mailer.py            SMTP from the bot's own mailbox
```

The report covers the **running month up to the morning of the run** (the
timer fires before the month ends) and compares it with the **previous,
complete month**. The text says so in its first sentence.

Current sections: parking (hour-of-day and weekend patterns per lot, month vs.
previous month, data quality). Weather, air quality and traffic follow the same
path: a module under `Statistical_models/` that returns a facts dict, added to
`report/facts.py`.

## Layout

```
Sleeping Imidillo/
├── config.py                 all settings, read once from .env; `python config.py` prints them masked
├── requirements.txt
├── deploy/
│   ├── imidillo-report.service      oneshot unit running jobs.monthly_report
│   ├── imidillo-report.timer        last Monday of the month, 10:00, Persistent
│   └── imidillo-report-test.timer   every 10 min, smoke test only
├── jobs/
│   └── monthly_report.py     the entry point: facts -> write -> verify -> render -> send
├── data/
│   ├── timewindow.py         report window (local month, capped at "now") <-> UTC
│   ├── quantumleap.py        sensor history, one entity + window -> DataFrame
│   ├── orion.py              current entity metadata (names, capacities)
│   ├── cleaning.py           dedupe, local time, hourly grid, coverage
│   └── parking.py            lots: capacity, occupancy %, usable / excluded verdict
├── Statistical_models/
│   ├── parking_hourly_regression.py   occ% ~ hour + weekend, Newey-West errors, per lot + pooled test
│   └── parking_comparison.py          month vs previous month at equal hour and weekday
├── report/
│   ├── facts.py              runs the models, bundles facts.json (+ data quality, glossary)
│   ├── llm.py                one door to the OpenAI Responses API (retries, strict JSON)
│   ├── writer.py             the writer prompt and call
│   ├── verifier.py           number tracing + verifier prompt + revise/cut loop
│   └── render.py             Markdown -> email HTML, charts (matplotlib), PDF (xhtml2pdf)
├── mailer.py                 build_message / send_message; --test and --send for manual checks
├── tests/smoke_test.py       every stage end to end; `--llm` adds the writer/verifier steps
├── static/avatar-standing.png
└── output/<month>/           facts.json, draft.md, report.md, verification.json, report.html,
                              report.pdf, chart_*.png   (gitignored)
```

## Setup

Python 3.11 or newer (developed on 3.14).

```bash
git clone <repo> && cd <repo>
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
cp .env.example .env                               # then fill it in, see below
.venv/bin/python config.py                         # prints what is set, masked; exit 1 if something is missing
```

### `.env`

| Variable | Needed for | Notes |
|---|---|---|
| `QL_BASE_URL`, `ORION_BASE_URL` | facts | IMIQ public gateway; defaults are correct, no api key required |
| `FIWARE_API_KEY` | optional | sent as `x-api-key` if set |
| `OPENAI_API_KEY` | write, verify | `WRITER_MODEL` / `VERIFIER_MODEL` default to `gpt-5.4` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM_NAME` | send | the bot's own mailbox; Gmail needs an app password with 2FA on |
| `REPORT_EMAIL_TO` | send | one address or several, comma-separated; everyone sees each other in To |
| `REPORT_LANGUAGE` | write | `en` (default), `de`, `tr` |
| `LLM_REASONING`, `LLM_TIMEOUT_S` | optional | reasoning effort (`medium`) and per-call timeout (180 s) |

Missing settings never crash a box: the job prints what is missing and exits
with code 2 (skipped, not failed).

## Tests

```bash
.venv/bin/python -m tests.smoke_test 2026-08          # ~30 s, no LLM: data, statistics, facts, render, mailer build
.venv/bin/python -m tests.smoke_test 2026-08 --llm    # + writer and verifier (tokens, ~3 min); the verifier
                                                      #   must catch two injected fabrications
.venv/bin/python -m jobs.monthly_report --dry-run     # the real job for the current month, everything except sending
.venv/bin/python mailer.py --test                     # one plain test email to the bot's OWN inbox
.venv/bin/python mailer.py --send 2026-08             # the rendered report to the bot's own inbox
```

Every module also runs on its own (`python -m data.parking 2026-08`,
`python -m report.facts 2026-08`, ...) and prints what it did.

## Deploy (Linux box with systemd)

```bash
# 1. code + venv, as an unprivileged user (the .service assumes /opt/sleeping-imidillo and user "imidillo")
sudo git clone <repo> /opt/sleeping-imidillo
sudo chown -R imidillo:imidillo /opt/sleeping-imidillo
cd /opt/sleeping-imidillo
sudo -u imidillo python3 -m venv .venv
sudo -u imidillo .venv/bin/pip install -r requirements.txt

# 2. settings
sudo -u imidillo cp .env.example .env && sudo -u imidillo nano .env     # fill in; chmod 600 .env
sudo -u imidillo .venv/bin/python config.py                             # must end with "All required settings present."

# 3. prove it runs on this box (no email)
sudo -u imidillo .venv/bin/python -m tests.smoke_test
sudo -u imidillo .venv/bin/python -m jobs.monthly_report --dry-run       # ~5 min, output/<month>/ appears

# 4. systemd units (edit WorkingDirectory/ExecStart/User in the .service first if the paths differ)
sudo cp deploy/imidillo-report.* /etc/systemd/system/
sudo systemctl daemon-reload
systemd-analyze verify /etc/systemd/system/imidillo-report.service
systemd-analyze calendar 'Mon *-*~07/1 10:00:00'                       # prints the next last-Mondays

# 5. smoke-test the SCHEDULING: the service fires every 10 minutes and sends the report
#    (to REPORT_EMAIL_TO! set it to your own address for this step)
sudo systemctl enable --now imidillo-report-test.timer
journalctl -u imidillo-report.service -f
# ...one or two runs later:
sudo systemctl disable --now imidillo-report-test.timer

# 6. arm the real schedule
sudo systemctl enable --now imidillo-report.timer
systemctl list-timers imidillo-report.timer                             # shows the NEXT run
```

`Persistent=true` on the timer means a missed run (box off at 10:00) fires as
soon as the machine is back. The job needs outbound HTTPS (IMIQ gateway,
OpenAI) and SMTP (port 587 or 465).

## Operations

- Exit codes of `jobs.monthly_report`: `0` sent, `2` not configured (skip),
  `1` failure with the reason in `journalctl -u imidillo-report.service`;
  `SuccessExitStatus=2` keeps a skip from showing as failed.
- Everything a run produced is in `output/<month>/`: `facts.json` (what the
  LLM saw), `draft.md`, `verification.json` (every claim and its verdict),
  `report.md` (approved text), `report.html`, `report.pdf`.
- Rebuild a past month: `python -m jobs.monthly_report 2026-08 --dry-run`.
  Re-run only the LLM steps on saved facts: `python -m report.writer 2026-08`
  then `python -m report.verifier 2026-08`.
- A run takes 4 to 6 minutes; almost all of it is the verifier model.
- Lots with less than `MIN_COVERAGE_PCT` (60 %, `data/cleaning.py`) of hours
  reported are excluded and listed in the report with the reason; raise it
  back to 70 once the sensors are more reliable.

## Troubleshooting

- `SMTP login rejected`: Gmail needs an app password (2FA on), not the account password.
- No mail, no error: check the bot mailbox's Sent folder and the recipients' spam folder; mark the first one as not spam.
- `[JOB] skip not configured`: `python config.py` lists what is missing.
- `verifier did not approve`: see `output/<month>/verification.json`; the run sent nothing, by design.
