"""Visible offline checks for Dataforsyningen provider failure outcomes."""

import io
import urllib.error
from email.message import Message
from unittest.mock import patch

from PIL import Image

from dynamic_functions.Terrain.dataforsyningen import (
    _fetch_metatile,
    _http_get,
    _no_coverage_kind,
)


class _Response:
    """Small context-manager response used without opening the network."""

    status = 200

    def __init__(self, payload: bytes, content_type: str = "image/jpeg") -> None:
        self._payload = payload
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://provider.invalid/wms",
        status,
        f"synthetic HTTP {status}",
        Message(),
        None,
    )


def _classified_http_error(status: int) -> dict:
    with (
        patch(
            "dynamic_functions.Terrain.dataforsyningen.urllib.request.urlopen",
            side_effect=_http_error(status),
        ) as urlopen,
        patch("dynamic_functions.Terrain.dataforsyningen.time.sleep") as sleep,
    ):
        payload, result = _http_get("https://provider.invalid/wms")
    return {
        "status": result["status"],
        "httpStatus": result["httpStatus"],
        "payloadAbsent": payload is None,
        "attempts": urlopen.call_count,
        "retryDelays": [call.args[0] for call in sleep.call_args_list],
    }


@visible
def dataforsyningen_failures(tile_id: str) -> dict:
    """Verify provider failures offline without network or database access."""

    http_cases = {
        "authentication": _classified_http_error(401),
        "rateLimited": _classified_http_error(429),
        "transient": _classified_http_error(503),
        "provider": _classified_http_error(418),
    }

    network_error = urllib.error.URLError("synthetic connection failure")
    with (
        patch(
            "dynamic_functions.Terrain.dataforsyningen.urllib.request.urlopen",
            side_effect=network_error,
        ) as urlopen,
        patch("dynamic_functions.Terrain.dataforsyningen.time.sleep") as sleep,
    ):
        network_payload, network = _http_get(
            "https://provider.invalid/wms"
        )
    network_attempts = urlopen.call_count
    network_delays = [call.args[0] for call in sleep.call_args_list]

    retry_response = _Response(b"provider-image")
    with (
        patch(
            "dynamic_functions.Terrain.dataforsyningen.urllib.request.urlopen",
            side_effect=[_http_error(429), retry_response],
        ) as urlopen,
        patch("dynamic_functions.Terrain.dataforsyningen.time.sleep") as sleep,
    ):
        retry_payload, retry = _http_get("https://provider.invalid/wms")
    retry_attempts = urlopen.call_count
    retry_delays = [call.args[0] for call in sleep.call_args_list]

    with patch(
        "dynamic_functions.Terrain.dataforsyningen._http_get",
        return_value=(
            b"not an image",
            {"httpStatus": 200, "contentType": "image/jpeg"},
        ),
    ):
        corrupt_payload, corrupt = _fetch_metatile(tile_id, "fixture-token")

    white = Image.new("RGB", (1024, 1024), (255, 255, 255))
    white_buffer = io.BytesIO()
    white.save(white_buffer, format="PNG")
    no_coverage = _no_coverage_kind(
        Image.open(io.BytesIO(white_buffer.getvalue())).convert("RGB")
    )

    expected = {
        "authentication": "authentication_error",
        "rateLimited": "rate_limited",
        "transient": "transient_error",
        "provider": "provider_error",
    }
    checks = {
        name: case["status"] == expected[name]
        for name, case in http_cases.items()
    }
    checks.update(
        {
            "rateLimitRetried": (
                http_cases["rateLimited"]["attempts"] == 3
                and http_cases["rateLimited"]["retryDelays"] == [1, 2]
            ),
            "network": (
                network_payload is None
                and network["status"] == "network_error"
                and network_attempts == 3
                and network_delays == [1, 2]
            ),
            "retryRecovered": (
                retry_payload == b"provider-image"
                and retry["httpStatus"] == 200
                and retry_attempts == 2
                and retry_delays == [1]
            ),
            "corrupt": (
                corrupt_payload is None
                and corrupt["status"] == "corrupt_response"
            ),
            "noCoverage": no_coverage == "white_fill",
        }
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(
            "Dataforsyningen failure checks failed: " + ", ".join(failed)
        )

    return {
        "tileId": tile_id,
        "offline": True,
        "databaseAccess": False,
        "checks": checks,
        "httpCases": http_cases,
        "network": {
            "status": network["status"],
            "attempts": network_attempts,
            "retryDelays": network_delays,
        },
        "retryRecovery": {
            "status": "success",
            "attempts": retry_attempts,
            "retryDelays": retry_delays,
        },
        "corruptStatus": corrupt["status"],
        "noCoverageStatus": no_coverage,
    }
