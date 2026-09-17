# Report time windows.
#
# The report covers one LOCAL calendar month (config.LOCAL_TZ); QuantumLeap
# takes UTC. This is the only module that converts between the two.
#
# The job runs on the LAST MONDAY of the month, so the report month is still
# running: report_window() caps the window at "now" (last full hour) and the
# facts carry period_label / days_covered so the text can say so. A finished
# month (the comparison month) gets its full window.

from __future__ import annotations
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import config

def month_window(year: int, month: int) -> tuple[datetime, datetime]:
    """UTC [start, end) of the full local calendar month."""
    tz = ZoneInfo(config.LOCAL_TZ)
    start_local = datetime(year, month, 1, tzinfo=tz)
    if month == 12:
        end_local = datetime(year + 1, 1, 1, tzinfo=tz)
    else:
        end_local = datetime(year, month + 1, 1, tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def report_window(year: int, month: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    """UTC [start, end) the report covers: the calendar month, capped at `now`
    floored to the full hour. A finished month gets its full window; the
    running month is reported up to its last complete hour."""
    start, end = month_window(year, month)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    now = now.replace(minute=0, second=0, microsecond=0)
    return start, min(end, max(now, start))


def period_label(start: datetime, end: datetime) -> str:
    """'1 to 28 September 2026' for a window (end exclusive), local dates."""
    tz = ZoneInfo(config.LOCAL_TZ)
    first = start.astimezone(tz)
    last = (end - timedelta(hours=1)).astimezone(tz)      # last covered hour
    if (first.year, first.month) == (last.year, last.month):
        return f"{first.day} to {last.day} {first.strftime('%B %Y')}"
    return f"{first.day} {first.strftime('%B')} to {last.day} {last.strftime('%B %Y')}"


def days_covered(start: datetime, end: datetime) -> int:
    """Number of local calendar days touched by the window (end exclusive)."""
    tz = ZoneInfo(config.LOCAL_TZ)
    first = start.astimezone(tz).date()
    last = (end - timedelta(hours=1)).astimezone(tz).date()
    return max(0, (last - first).days + 1)


def current_month(today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    return today.year, today.month


def previous_month(today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1

def previous_of(year: int, month: int) -> tuple[int, int]:
    """(year, month) of the month before the given one: (2026, 1) -> (2025, 12)."""
    return (year - 1, 12) if month == 1 else (year, month - 1)

def parse_month(text: str) -> tuple[int, int]:
    year, month = text.split("-")
    year_i, month_i = int(year), int(month)
    if not 1 <= month_i <= 12:
        raise ValueError(f"month out of range: {text}")
    return year_i, month_i


def month_label(year: int, month: int) -> str:
    return date(year, month, 1).strftime("%B %Y")

if __name__ == "__main__":
    y, m = current_month()
    s, e = report_window(y, m)
    print(f"report month   : {month_label(y, m)}  ->  covers {period_label(s, e)} ({days_covered(s, e)} days)")
    print(f"UTC window     : {s.isoformat()}  ->  {e.isoformat()}  (end exclusive)")
    py, pm = previous_of(y, m)
    ps, pe = month_window(py, pm)
    print(f"compared with  : {month_label(py, pm)}  ->  {period_label(ps, pe)} ({days_covered(ps, pe)} days)")