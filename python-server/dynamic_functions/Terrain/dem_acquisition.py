"""Explicit multi-provider DEM acquisition and selection."""

from __future__ import annotations

import logging

import numpy as np

from dynamic_functions.Terrain.arctic_dem import (
    _fetch_heightmap as _fetch_arcticdem,
)
from dynamic_functions.Terrain.copernicus_dem import (
    _fetch_heightmap as _fetch_copernicus,
)


log = logging.getLogger("terrain.dem_acquisition")


class DemProviderNoDataError(RuntimeError):
    """A DEM provider completed but returned no finite samples."""


class DemAcquisitionError(RuntimeError):
    """No configured DEM provider produced usable data."""

    def __init__(self, tile_id: str, failures: list[Exception]) -> None:
        self.tile_id = tile_id
        self.failures = tuple(failures)
        details = "; ".join(str(failure) for failure in failures)
        super().__init__(f"DEM acquisition failed tile={tile_id}: {details}")


def _valid_samples(heightmap: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(heightmap)))


def _provider_failure(provider: str, tile_id: str, exc: Exception) -> Exception:
    failure = RuntimeError(
        f"provider={provider} tile={tile_id} {type(exc).__name__}: {exc}"
    )
    failure.__cause__ = exc
    return failure


def fetch_best_dem(tile_id: str) -> dict:
    """Acquire both DEM providers and select the most complete result."""

    candidates: list[dict] = []
    attempts: list[dict] = []
    failures: list[Exception] = []

    try:
        heightmap, sources, geoid_undulation = _fetch_arcticdem(tile_id)
        valid_samples = _valid_samples(heightmap)
        if valid_samples == 0:
            raise DemProviderNoDataError(
                f"provider=arcticdem tile={tile_id} returned no finite samples"
            )
        candidates.append(
            {
                "heightmap": heightmap,
                "source": "arcticdem_10m",
                "provider": "arcticdem",
                "dataset": "mosaics/v4.1/10m",
                "verticalDatum": "EGM2008",
                "geoidUndulation": geoid_undulation,
                "sources": sources,
                "validSamples": valid_samples,
            }
        )
        attempts.append(
            {
                "provider": "arcticdem",
                "status": "ready",
                "validSamples": valid_samples,
                "sourceCount": len(sources),
            }
        )
    except Exception as exc:
        failures.append(_provider_failure("arcticdem", tile_id, exc))
        attempts.append(
            {
                "provider": "arcticdem",
                "status": "failed",
                "errorType": type(exc).__name__,
                "error": str(exc),
            }
        )
        log.warning("ArcticDEM acquisition failed tile=%s: %s", tile_id, exc)

    try:
        heightmap, sources = _fetch_copernicus(tile_id)
        valid_samples = _valid_samples(heightmap)
        if valid_samples == 0:
            raise DemProviderNoDataError(
                f"provider=copernicus tile={tile_id} returned no finite samples"
            )
        candidates.append(
            {
                "heightmap": heightmap,
                "source": "copernicus",
                "provider": "copernicus",
                "dataset": "Copernicus DEM GLO-30",
                "verticalDatum": "EGM2008",
                "geoidUndulation": None,
                "sources": sources,
                "validSamples": valid_samples,
            }
        )
        attempts.append(
            {
                "provider": "copernicus",
                "status": "ready",
                "validSamples": valid_samples,
                "sourceCount": len(sources),
            }
        )
    except Exception as exc:
        failures.append(_provider_failure("copernicus", tile_id, exc))
        attempts.append(
            {
                "provider": "copernicus",
                "status": "failed",
                "errorType": type(exc).__name__,
                "error": str(exc),
            }
        )
        log.warning("Copernicus acquisition failed tile=%s: %s", tile_id, exc)

    if not candidates:
        raise DemAcquisitionError(tile_id, failures)

    # Candidate order is the tie-breaker: ArcticDEM retains its 10 m advantage.
    selected = max(candidates, key=lambda candidate: candidate["validSamples"])
    log.info(
        "DEM selected provider=%s tile=%s valid_samples=%d attempts=%s",
        selected["provider"],
        tile_id,
        selected["validSamples"],
        ",".join(
            f"{attempt['provider']}:{attempt['status']}"
            for attempt in attempts
        ),
    )
    return {**selected, "attempts": attempts}
