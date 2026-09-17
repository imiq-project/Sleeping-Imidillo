
# QuantumLeap client: sensor history for one entity and one time window

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pandas as pd
import requests

import config
from data.timewindow import month_window, parse_month

PAGE_SIZE = 10_000    
_RETRIES = 3
_BACKOFF_S = 2.0

class QuantumLeapError(RuntimeError):
    """QuantumLeap unreachable or returned an error we cannot interpret."""


def _headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if config.FIWARE_API_KEY:
        h["x-api-key"] = config.FIWARE_API_KEY
    if config.FIWARE_SERVICE:
        h["Fiware-Service"] = config.FIWARE_SERVICE
    if config.FIWARE_SERVICE_PATH:
        h["Fiware-ServicePath"] = config.FIWARE_SERVICE_PATH
    return h


def _get(path: str, params: dict | None = None) -> requests.Response | None:
    """GET with retries. Returns None on 404 (= no data), raises on other failures."""
    url = f"{config.QL_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    last_error: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(),
                                timeout=config.QL_TIMEOUT_S)
        except requests.RequestException as e:
            last_error = e
        else:
            if resp.status_code == 404:
                return None
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
            last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if attempt < _RETRIES:
            time.sleep(_BACKOFF_S * attempt)
    raise QuantumLeapError(f"GET {url} failed after {_RETRIES} attempts: {last_error}")


def list_entities() -> pd.DataFrame:
    """All entities known to QuantumLeap: columns entityId, entityType, last_seen (UTC)."""
    resp = _get("entities")
    rows = resp.json() if resp is not None else []
    df = pd.DataFrame(rows, columns=["entityId", "entityType", "index"])
    df = df.rename(columns={"index": "last_seen"})
    df["last_seen"] = pd.to_datetime(df["last_seen"], utc=True)
    return df.sort_values(["entityType", "entityId"]).reset_index(drop=True)


def _page_to_frame(payload: dict) -> pd.DataFrame:
    index = pd.to_datetime(payload.get("index", []), utc=True)
    columns = {a["attrName"]: a["values"] for a in payload.get("attributes", [])}
    df = pd.DataFrame(columns, index=index)
    df.index.name = "timestamp"
    for col in df.columns:
        # Numbers arrive as floats; strings (e.g. status) stay strings.
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= df[col].notna().sum():
            df[col] = converted
    return df


def fetch_history(entity_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Raw history of one entity in [start, end), both tz-aware datetimes.

    Pages through QuantumLeap's 10 000-row cap so a busy sensor's month
    arrives complete. Rows are returned in the order stored, unsorted and
    un-deduplicated on purpose (see module docstring).
    """
    params = {
        "fromDate": start.isoformat().replace("+00:00", "Z"),
        "toDate": end.isoformat().replace("+00:00", "Z"),
        "limit": PAGE_SIZE,
        "offset": 0,
    }
    pages: list[pd.DataFrame] = []
    while True:
        resp = _get(f"entities/{entity_id}", params)
        if resp is None:            # 404 = nothing stored in this window
            break
        page = _page_to_frame(resp.json())
        pages.append(page)
        if len(page) < PAGE_SIZE:
            break
        params["offset"] += PAGE_SIZE
    if not pages:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    return pd.concat(pages)


def fetch_type(entity_type: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """History of every entity of one type, keyed by entityId. Entities with
    no rows in the window are still present, mapped to an empty frame, so the
    coverage step can report them as silent."""
    ids = list_entities().query("entityType == @entity_type")["entityId"]
    return {eid: fetch_history(eid, start, end) for eid in ids}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe QuantumLeap.")
    parser.add_argument("entity_id", nargs="?", help="e.g. ParkingSpot:Ulrichshaus")
    parser.add_argument("month", nargs="?", help="YYYY-MM, e.g. 2026-08")
    parser.add_argument("--list", action="store_true", help="inventory of entity types")
    args = parser.parse_args()

    if not config.quantumleap_configured():
        print("[QL] QL_BASE_URL not set in .env")
        return 2

    if args.list or not args.entity_id:
        ents = list_entities()
        print(f"{len(ents)} entities at {config.QL_BASE_URL}")
        for etype, group in ents.groupby("entityType"):
            print(f"  {etype:18s} {len(group):3d}   e.g. {', '.join(group['entityId'].head(3))}")
        return 0

    year, month = parse_month(args.month)
    start, end = month_window(year, month)
    print(f"[QL] {args.entity_id}  {start.isoformat()} -> {end.isoformat()}")
    df = fetch_history(args.entity_id, start, end)
    if df.empty:
        print("[QL] no rows in this window")
        return 0
    print(f"[QL] {len(df)} raw rows, {df.index.nunique()} distinct timestamps")
    print(f"[QL] first {df.index.min()}  last {df.index.max()}")
    print(f"[QL] columns: {list(df.columns)}")
    print(df.head(5).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())