"""Coastal viewer demand for contract-depth Glacier bathymetry jobs."""

from __future__ import annotations

import math
import os
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np

from dynamic_functions.Terrain.Database.database import DATABASE_PATH
from dynamic_functions.Terrain.terrain_config import (
    GREENLAND_BBOX,
    WMS_CONTRACT_DEPTH,
)
from dynamic_functions.Terrain.tile_address import format_tile_id, parse_tile_id


BATHYMETRY_JOB_DEPTH = 8
OFFSHORE_LIMIT_M = 2_000.0


class BathymetryDeferredError(OSError):
    """A Glacier job whose independently demanded terrain is still settling."""


def _ancestor_at_depth(tile_id: str, depth: int) -> tuple[int, int] | None:
    parsed = parse_tile_id(tile_id)
    if parsed is None:
        return None
    source_depth, column, row = parsed
    if source_depth < depth:
        return None
    shift = source_depth - depth
    return column >> shift, row >> shift


def _decode_mask(width: int, height: int, blob) -> np.ndarray | None:
    values = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    if values.size != int(width) * int(height):
        return None
    return values.reshape((int(height), int(width))).astype(bool)


def _tile_gap_m(
    left: tuple[int, int],
    right: tuple[int, int],
    tile_size_m: float,
) -> float:
    column_gap = max(abs(left[0] - right[0]) - 1, 0)
    row_gap = max(abs(left[1] - right[1]) - 1, 0)
    return math.hypot(column_gap, row_gap) * tile_size_m


def eligible_fjord_jobs(
    connection: sqlite3.Connection,
    visible_tile_ids: Iterable[str],
    *,
    offshore_limit_m: float = OFFSHORE_LIMIT_M,
) -> set[str]:
    """Return missing depth-8 jobs justified by visible coastal water."""

    candidates = {
        address
        for tile_id in visible_tile_ids
        if (address := _ancestor_at_depth(tile_id, WMS_CONTRACT_DEPTH))
        is not None
    }
    if not candidates:
        return set()

    root_width = float(GREENLAND_BBOX[2] - GREENLAND_BBOX[0])
    tile_size_m = root_width / (1 << WMS_CONTRACT_DEPTH)
    search_tiles = int(math.ceil(offshore_limit_m / tile_size_m)) + 1
    minimum_column = min(column for column, _ in candidates) - search_tiles
    maximum_column = max(column for column, _ in candidates) + search_tiles
    minimum_row = min(row for _, row in candidates) - search_tiles
    maximum_row = max(row for _, row in candidates) + search_tiles

    masks: dict[tuple[int, int], tuple[bool, bool]] = {}
    rows = connection.execute(
        "SELECT t.col,t.row,m.width,m.height,m.mask "
        "FROM coastline_masks m JOIN tiles t ON t.tile_id=m.tile_id "
        "WHERE t.depth=? AND t.col BETWEEN ? AND ? "
        "AND t.row BETWEEN ? AND ?",
        (
            WMS_CONTRACT_DEPTH,
            minimum_column,
            maximum_column,
            minimum_row,
            maximum_row,
        ),
    ).fetchall()
    for column, row, width, height, blob in rows:
        mask = _decode_mask(width, height, blob)
        if mask is not None:
            masks[(int(column), int(row))] = (
                bool(np.any(mask)),
                bool(np.all(mask)),
            )

    mixed = {
        address
        for address, (has_water, all_water) in masks.items()
        if has_water and not all_water
    }
    if not mixed:
        return set()

    eligible: set[tuple[int, int]] = set()
    for address in candidates:
        classification = masks.get(address)
        if classification is None or not classification[0]:
            continue
        if not classification[1] or any(
            _tile_gap_m(address, coast, tile_size_m) <= offshore_limit_m
            for coast in mixed
        ):
            eligible.add(address)

    jobs = {
        format_tile_id(
            BATHYMETRY_JOB_DEPTH,
            column >> (WMS_CONTRACT_DEPTH - BATHYMETRY_JOB_DEPTH),
            row >> (WMS_CONTRACT_DEPTH - BATHYMETRY_JOB_DEPTH),
        )
        for column, row in eligible
    }
    if not jobs:
        return set()
    marks = ",".join("?" for _ in jobs)
    covered = {
        row[0]
        for row in connection.execute(
            f"SELECT tile_id FROM bathymetry WHERE tile_id IN ({marks})",
            sorted(jobs),
        )
    }
    return jobs - covered


def run_bathymetry_job(job_id: str) -> dict:
    """Run Glacier's idempotent contract-depth worker for one fjord region."""

    root = Path(
        os.environ.get("GLACIER_ROOT", str(Path.home() / "work" / "glacier"))
    ).expanduser().resolve()
    command = root / "runOnDemand"
    if not command.is_file():
        raise RuntimeError(f"Glacier worker is missing: {command}")
    base = os.environ.get(
        "TERRAIN_VIEWER_BASE", "http://localhost:5180"
    ).strip()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "SERVER_DIR": str(DATABASE_PATH.parents[3]),
        }
    )
    completed = subprocess.run(
        [
            str(command),
            "--tile",
            job_id,
            "--db",
            str(DATABASE_PATH),
            "--python",
            sys.executable,
            "--base",
            base,
            "--commit",
        ],
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stderr or completed.stdout or "").strip()
    if completed.returncode:
        if "coverage incomplete" in output.lower():
            raise BathymetryDeferredError(
                output.splitlines()[-1] if output else "coverage incomplete"
            )
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return {
        "tileId": job_id,
        "written": True,
        "worker": str(command),
        "summary": output.splitlines()[-1] if output else "completed",
    }
