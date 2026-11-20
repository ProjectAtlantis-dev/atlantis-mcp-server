import atlantis
import logging

logger = logging.getLogger("dynamic_function")


@visible
async def myTableWithDates() -> list[dict[str, object]]:
    """Return a fictional shipping log of a package travelling from Tokyo, Japan to Tallinn, Estonia"""
    # times are ISO 8601 strings in each facility's local time zone, so the offset shows where the scan happened
    # the delivery row has no time_out because the package does not leave the recipient
    rows = [
        {"step": 1, "facility": "Shinagawa Pickup Center", "city": "Tokyo, Japan", "time_in": "2026-11-07T10:30:00+09:00", "time_out": "2026-11-07T18:15:00+09:00", "status": "Received at origin facility"},
        {"step": 2, "facility": "Narita International Export Hub", "city": "Narita, Japan", "time_in": "2026-11-07T20:40:00+09:00", "time_out": "2026-11-08T17:55:00+09:00", "status": "Export customs cleared"},
        {"step": 3, "facility": "Pacific Gateway Air Hub", "city": "Los Angeles, USA", "time_in": "2026-11-08T11:30:00-08:00", "time_out": "2026-11-08T23:10:00-08:00", "status": "Transited in bond"},
        {"step": 4, "facility": "Riverbend Air Hub", "city": "Cincinnati, USA", "time_in": "2026-11-09T07:55:00-05:00", "time_out": "2026-11-09T19:30:00-05:00", "status": "Sorted"},
        {"step": 5, "facility": "Halle Gateway Sort Center", "city": "Leipzig, Germany", "time_in": "2026-11-10T11:05:00+01:00", "time_out": "2026-11-10T23:40:00+01:00", "status": "EU import customs cleared"},
        {"step": 6, "facility": "Baltic Crossdock", "city": "Riga, Latvia", "time_in": "2026-11-11T03:30:00+02:00", "time_out": "2026-11-11T10:15:00+02:00", "status": "Transferred to regional truck"},
        {"step": 7, "facility": "Ülemiste Delivery Station", "city": "Tallinn, Estonia", "time_in": "2026-11-11T14:40:00+02:00", "time_out": "2026-11-12T08:30:00+02:00", "status": "Sorted for local delivery"},
        {"step": 8, "facility": "Recipient address", "city": "Tallinn, Estonia", "time_in": "2026-11-12T11:47:00+02:00", "time_out": None, "status": "Delivered"},
    ]
    # elapsed repeats each scan's time_in; the 'when' formatter has the client render it relative to now (e.g. "8 days ago")
    for row in rows:
        row["elapsed"] = row["time_in"]
    await atlantis.client_data("Shipping log", rows, column_formatter={
        "elapsed": {"type": "when"},
    })
    return rows
