# Sleeping Imidillo

Imidillo is the Magdeburg city assistant of the IMIQ project. This repository is
its "other job": once a month, unattended, it emails the project team a short
city report built from our digital twin. A systemd timer wakes
one Python script, the script does its work and exits.

The statistics are computed deterministically, an LLM turns the numbers into prose and a second LLM checks
every sentence against those numbers before anything is sent. If the data cannot
support a claim, the claim is not in the report.

The report covers the running month up to the morning of the run (the
timer fires before the month ends) and compares it with the previous,
complete monthS. The text says so in its first sentence.

Current sections: parking (hour-of-day and weekend patterns per lot, month vs.
previous month, data quality). Weather, air quality and traffic follow the same
path: a module under `Statistical_models/` that returns a facts dict, added to
`report/facts.py`.
