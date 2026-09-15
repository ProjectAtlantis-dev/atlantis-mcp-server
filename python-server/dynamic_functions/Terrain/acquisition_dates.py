"""Acquisition-time provenance, independent of database publication times."""

from collections.abc import Iterable
from datetime import datetime, timezone
import json


def normalize_date(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    parsed = datetime.fromisoformat(value)
    # Provider timeutc and GTK50 timestamps without an offset are UTC.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def date_range(values: Iterable[str | None], *, source: str, scope: str) -> dict:
    dates = [normalize_date(value) for value in values]
    known = [value for value in dates if value is not None]
    return {
        "date": min(known, key=datetime.fromisoformat) if known else None,
        "dateEnd": max(known, key=datetime.fromisoformat) if known else None,
        "dateSource": source,
        "dateScope": scope,
        "dateComplete": bool(dates) and len(known) == len(dates),
    }


def encode_dates(value: dict | None) -> str | None:
    if value is None:
        return None
    # Validate before touching storage; never persist a made-up publication date.
    start = normalize_date(value["date"])
    end = normalize_date(value["dateEnd"])
    if (start is not None and end is not None
            and datetime.fromisoformat(start) > datetime.fromisoformat(end)):
        raise ValueError("acquisition date range is reversed")
    return json.dumps({**value, "date": start, "dateEnd": end}, sort_keys=True)


def decode_dates(value: str | None) -> dict:
    return json.loads(value) if value is not None else {"date": None, "dateEnd": None}
