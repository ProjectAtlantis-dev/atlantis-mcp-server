"""Acquire Asiaq assets from viewer coordinates into the MCP-owned catalog."""
from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import urllib.request
import zipfile

import atlantis
from pyproj import Transformer

from dynamic_functions.Terrain.Asset import database as assets
from dynamic_functions.Terrain.Asset import rebuild
from dynamic_functions.Terrain.Asset.settlements import SETTLEMENTS
from dynamic_functions.Terrain.Database import database as terrain
from dynamic_functions.Terrain.Database.tiles import (
    GRID_N, _decompress_float32, _decompress_uint8, write_dem,
)
from dynamic_functions.Terrain.coords import to_stereo
from dynamic_functions.Terrain.dem_acquisition import fetch_best_dem
from dynamic_functions.Terrain.demand import DemandLane
from dynamic_functions.Terrain.terrain_config import GREENLAND_BBOX
from dynamic_functions.Terrain.tile_address import tile_bounds

_HERE = Path(__file__).resolve().parent
_REGISTRY_KEY = 'Terrain.Asset.coordinate_demand.v1'
_INIT_LOCK = threading.Lock()
# The full source catalog is a spatial lookup, never a configured-town allowlist.
_CENTRES = tuple((folder, *to_stereo(lat, lon)) for folder, lat, lon in SETTLEMENTS)
_FOLDERS = {folder for folder, _, _ in SETTLEMENTS}
# Include town outskirts even when the catalog centre is outside the query box.
_SETTLEMENT_PAD_M = 5000.0
_GROUND_DEPTH = 12


def nearby_settlements(qx: float, qy: float, max_range: float) -> list[str]:
    if not all(math.isfinite(value) for value in (qx, qy, max_range)) or max_range <= 0:
        raise ValueError('building coordinates and range must be finite; range must be positive')
    radius = max_range + _SETTLEMENT_PAD_M
    return [folder for distance, folder in sorted(
        (math.hypot(cx - qx, cy - qy), folder)
        for folder, cx, cy in _CENTRES
        if abs(cx - qx) <= radius and abs(cy - qy) <= radius
    )]


def settlement_loaded(connection: sqlite3.Connection, code: str) -> bool:
    counts = dict(connection.execute(
        'SELECT type, COUNT(*) FROM assets WHERE substr(id,1,?)=? GROUP BY type',
        (len(code) + 1, code + '_'),
    ))
    has_receipts = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='asset_imports'"
    ).fetchone()
    receipt = connection.execute(
        'SELECT layer_counts FROM asset_imports WHERE settlement=?', (code,),
    ).fetchone() if has_receipts else None
    if receipt:
        return all(counts.get(layer, 0) == count for layer, count in json.loads(receipt[0]).items())
    # Migrated catalogs predate receipts. A roads-only import is not complete.
    return bool(counts.get('BYGNING') and (counts.get('VEJMIDTE') or counts.get('STIMIDTE')))


def _download_archive(folder: str) -> Path:
    if folder not in _FOLDERS:
        raise ValueError(f'unknown Asiaq settlement: {folder}')
    directory = _HERE / 'grundkort'
    directory.mkdir(exist_ok=True)
    target = directory / f'{folder.split("_")[0]}_TekniskGrundkort_SHP.zip'
    if target.is_file():
        return target
    partial = target.with_suffix('.zip.part')
    request = urllib.request.Request(
        f'https://kortforsyning.asiaq.gl/files/{folder}/SHP/{target.name}',
        headers={'User-Agent': 'atlantis-terrain/coordinate-assets'},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response, partial.open('wb') as out:
            size = 0
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise ValueError('Asiaq archive exceeds 512 MiB')
                out.write(chunk)
        with zipfile.ZipFile(partial) as archive:
            members = rebuild._archive_members(archive)
            for suffix in ('SHP', 'DBF', 'PRJ'):
                rebuild._required_member(archive, members, f'BYGNING.{suffix}')
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _read_ground_tile(tile_id: str):
    with terrain.connection_lock():
        row = terrain.db().execute(
            'SELECT heightmap,confidence_map FROM tiles WHERE tile_id=?', (tile_id,),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    if row[1] is None:
        raise ValueError(f'measured terrain tile {tile_id} has no confidence map')
    return _decompress_float32(row[0]), _decompress_uint8(row[1])


def _ground_tile(tile_id: str):
    cached = _read_ground_tile(tile_id)
    if cached is not None:
        return cached
    acquisition = fetch_best_dem(tile_id)
    with terrain.connection_lock():
        # Terrain's own demand may have published during provider I/O.
        if _read_ground_tile(tile_id) is None:
            write_dem(terrain.db(), tile_id, acquisition['heightmap'],
                      acquisition['source'], acquisition['verticalDatum'],
                      acquisition_dates=acquisition['acquisitionDates'])
    return _read_ground_tile(tile_id)


def _ground_samples(archive, members, code, transformer) -> dict[str, float]:
    attributes = rebuild._read_dbf_records(archive.read(members['BYGNING.DBF']))
    rings = list(rebuild._polygonz_outer_rings(archive.read(members['BYGNING.SHP'])))
    if len(attributes) != len(rings):
        raise ValueError(f'{code}: building DBF/shape count mismatch')
    samples, tiles = {}, {}
    root_x, root_y, root_x1, root_y1 = GREENLAND_BBOX
    for index, ring in enumerate(rings):
        if not ring or len(ring) < 3:
            continue
        if ring[0][:2] == ring[-1][:2]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        xs, ys, _ = zip(*ring)
        tx, ty = transformer.transform(xs, ys)
        cx, cy = sum(tx) / len(tx), sum(ty) / len(ty)
        col = math.floor((cx - root_x) / (root_x1 - root_x) * (1 << _GROUND_DEPTH))
        row = math.floor((cy - root_y) / (root_y1 - root_y) * (1 << _GROUND_DEPTH))
        tile_id = f'{_GROUND_DEPTH}-{col}-{row}'
        bounds = tile_bounds(tile_id, GREENLAND_BBOX)
        if tile_id not in tiles:
            tiles[tile_id] = _ground_tile(tile_id)
        heightmap, confidence = tiles[tile_id]
        ix = int((cx - bounds[0]) / (bounds[2] - bounds[0]) * (GRID_N - 1))
        iy = int((cy - bounds[1]) / (bounds[3] - bounds[1]) * (GRID_N - 1))
        ground = float(heightmap[iy, ix])
        asset_id = f'{code}_{attributes[index].get("lokal_id", "") or index}'
        if confidence[iy, ix] == 0 or not math.isfinite(ground):
            raise ValueError(f'{asset_id}: no measured terrain at building centre ({tile_id})')
        samples[asset_id] = ground
    return samples


def _write_json(path: Path, value, *, compact: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=compact,
            separators=(',', ':') if compact else None,
            indent=None if compact else 2,
        ) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_settlement(folder: str) -> dict:
    """Fetch and import one complete package; network work stays off HTTP threads."""
    archive_path = _download_archive(folder)
    code = rebuild._settlement_code(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = rebuild._archive_members(archive)
        prj = rebuild._required_member(archive, members, 'BYGNING.PRJ').decode('latin1')
        transformer = Transformer.from_crs(rebuild._source_epsg(prj), 3413, always_xy=True)
        samples = _ground_samples(archive, members, code, transformer)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with assets.connection_lock():
            connection = assets.db()
            connection.execute(
                'CREATE TABLE IF NOT EXISTS asset_imports ('
                'settlement TEXT PRIMARY KEY, layer_counts TEXT NOT NULL, updated_at TEXT NOT NULL)'
            )
            enabled = dict(connection.execute(
                'SELECT id,enabled FROM assets WHERE substr(id,1,?)=?',
                (len(code) + 1, code + '_'),
            ))
            try:
                connection.execute(
                    "DELETE FROM assets WHERE substr(id,1,?)=? AND type IN ('BYGNING','VEJMIDTE','STIMIDTE')",
                    (len(code) + 1, code + '_'),
                )
                counts = {'BYGNING': rebuild._ingest_buildings(
                    connection, archive, members, code, transformer, samples, now,
                )}
                counts.update(rebuild._ingest_roads(connection, archive, members, code, transformer, now))
                connection.executemany('UPDATE assets SET enabled=? WHERE id=?',
                    [(value, asset_id) for asset_id, value in enabled.items()])
                rebuild.record_import(connection, code, counts, now)
                # This is rebuild inventory, automatically extended by acquisition.
                # Persist source inputs before marking the DB import complete.
                metadata_path = _HERE / 'metadata.json'
                samples_path = _HERE / 'building_ground_samples.json'
                metadata = rebuild._read_metadata(metadata_path)
                registry = json.loads(samples_path.read_text())
                rebuild._read_ground_samples(samples_path)
                registry['samples'].update(samples)
                metadata['grundkortSettlements'] = sorted(set(metadata['grundkortSettlements']) | {code})
                _write_json(samples_path, registry, compact=True)
                _write_json(metadata_path, metadata)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
    return {'settlement': code, 'layerCounts': counts,
            'archiveSha256': hashlib.sha256(archive_path.read_bytes()).hexdigest()}


def _lane() -> DemandLane:
    with _INIT_LOCK:
        lane = atlantis.server_shared.get(_REGISTRY_KEY)
        if lane is None:
            lane = DemandLane('assets', acquire_settlement, 1)
            atlantis.server_shared.set(_REGISTRY_KEY, lane)
        return lane


def request_for_point(qx: float, qy: float, max_range: float) -> dict:
    folders = nearby_settlements(qx, qy, max_range)
    # All database access is bounded local work. Providers run in the lane.
    with assets.connection_lock():
        connection = assets.db()
        missing = [folder for folder in folders
                   if not settlement_loaded(connection, folder.split('_')[0])]
    state = _lane().replace_pending(missing)
    failures = {folder: failure for folder, failure in state['failures'].items() if folder in missing}
    pending = bool(state['claimedActiveCount'] or state['pendingCount']
                   or any(failure.get('retryable') for failure in failures.values()))
    return {'settlements': missing, 'failures': failures, 'shouldPoll': pending,
            'status': 'loading' if pending else ('error' if failures else 'ready')}
