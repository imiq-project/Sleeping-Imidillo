# Sleeping Imidillo

Imidillo is the Magdeburg city assistant of the IMIQ project. This repository is its "other job": once a month, unattended, it emails the project team a short city report built from our digital twin. A systemd timer wakes one Python script, the script does its work and exits.

The statistics are computed deterministically, an LLM turns the numbers into prose and a second LLM checks every sentence against those numbers before anything is sent. If the data cannot support a claim, the claim is not in the report.

The report covers the running month up to the morning of the run (the timer fires before the month ends) and compares it with the previous, complete months. The text says so in its first sentence.

Current sections: parking, EV charging, traffic, and air quality (NO2, O3, PM10, PM2.5) — hour-of-day and weekend patterns per entity, month vs. previous month, data quality, and EU threshold exceedances for air quality. Weather, river level and building energy follow the same path: a module under `Statistical_models/` that returns a facts dict, added to `report/facts.py`.

*The Model Families*

**1- Rhythm model :** Hourly value regressed on hour-of-day and a weekend flag, with autocorrelation-robust standard errors. It answers: what is the typical day, when is the peak, how different are weekends, did any of that change since last month. Same model, different variable, for parking occupancy, traffic counts, pollutants and EV charger occupancy. EV data is event-driven (a record only when a car arrives or leaves), so the state is carried forward between records before the hourly values are taken. The same idea on daily instead of hourly data covers building energy: 27 campus buildings report electricity once per day (9 also heating), so the model compares weekdays with weekends and this month with the last.

**2- Summary and threshold model:** Monthly descriptives (mean, extremes, percentiles), counts of threshold exceedances (EU air-quality limits, flood alert levels), and a proper month-on-month test. For weather, air-quality limits and river level.

**3- Cross-sensor relationships:** The "correlations" across data entities. Raw hourly correlations are misleading because every sensor shares the same daily cycle, so we first remove the hour and weekday effects and correlate the residuals, with lags. Only significant relationships go into the report.

**Status:**
✅ Parking, traffic, air quality (rhythm + thresholds), EV charger occupancy — done
📋 Planned: building energy, weather, river level, cross-sensor correlations, events

*Too optimistic for now but included in the plan*

**4- Forecasting Models**
