# Orion client: the CURRENT state of entities (no history).
from __future__ import annotations

import sys

import requests

import config

_TIMEOUT_S = 30


class OrionError(RuntimeError):
    """Orion unreachable or returned an error."""


def _headers() -> dict[str, str]:
    h = {"Accept": "application/json"}
    if config.FIWARE_API_KEY:
        h["x-api-key"] = config.FIWARE_API_KEY
    if config.FIWARE_SERVICE:
        h["Fiware-Service"] = config.FIWARE_SERVICE
    if config.FIWARE_SERVICE_PATH:
        h["Fiware-ServicePath"] = config.FIWARE_SERVICE_PATH
    return h


def list_entities(entity_type: str) -> list[dict]:
    """Current attributes of every entity of one type, as flat dicts."""
    url = f"{config.ORION_BASE_URL.rstrip('/')}/entities"
    params = {"type": entity_type, "options": "keyValues", "limit": 1000}
    try:
        resp = requests.get(url, params=params, headers=_headers(), timeout=_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OrionError(f"GET {url} type={entity_type}: {e}") from e
    data = resp.json()
    if not isinstance(data, list):
        raise OrionError(f"unexpected response shape: {str(data)[:200]}")
    return data


def main() -> int:
    entity_type = sys.argv[1] if len(sys.argv) > 1 else "Parking"
    ents = list_entities(entity_type)
    print(f"{len(ents)} {entity_type} entities in Orion")
    for e in sorted(ents, key=lambda x: x["id"]):
        extras = {k: v for k, v in e.items() if k not in ("id", "type", "location")}
        print(f"  {e['id']:40s} {extras}")
    return 0


if __name__ == "__main__":
    sys.exit(main())