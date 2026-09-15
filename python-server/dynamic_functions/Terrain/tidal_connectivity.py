"""Read-only same-depth derivation of tidally connected WMS hydrography."""

from __future__ import annotations

import datetime
import hashlib
import sqlite3
import zlib
from typing import cast

import numpy as np
from scipy.ndimage import label

from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.tile_address import require_tile_id


SOURCE = "derived_tidal_connectivity"
VERSION = 1
SEA_SEED_MAX_ELEV_M = 0.5


def write_connectivity_snapshot(
    connection: sqlite3.Connection,
    tile_id: str,
    mask: np.ndarray,
    *,
    commit: bool = True,
) -> bool:
    """Publish an already-derived connectivity mask atomically.

    This is deliberately not a visible tool. The future background lane owns
    snapshot production; the interactive composition path may only read rows
    that have already been published.
    """

    require_tile_id(tile_id)
    values = np.asarray(mask)
    if values.ndim != 2 or not values.size:
        raise ValueError(
            "tidal connectivity mask must be a non-empty 2D array"
        )
    canonical = values.astype(np.uint8)
    if not np.all((canonical == 0) | (canonical == 1)):
        raise ValueError(
            "tidal connectivity mask must contain only boolean values"
        )
    encoded = zlib.compress(canonical.tobytes(), level=6)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO tidal_connectivity_masks "
        "(tile_id,width,height,mask,source,version,updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            tile_id,
            int(canonical.shape[1]),
            int(canonical.shape[0]),
            encoded,
            SOURCE,
            VERSION,
            now,
        ),
    )
    if cursor.rowcount == 1:
        if commit:
            connection.commit()
        return True
    row = connection.execute(
        "SELECT width,height,mask,source,version "
        "FROM tidal_connectivity_masks WHERE tile_id=?",
        (tile_id,),
    ).fetchone()
    expected = (
        int(canonical.shape[1]),
        int(canonical.shape[0]),
        encoded,
        SOURCE,
        VERSION,
    )
    if row == expected:
        return False
    raise RuntimeError(
        f"Refusing to clobber tidal connectivity snapshot {tile_id}"
    )


def read_connectivity_snapshot(
    connection: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read one ready connectivity snapshot without deriving or writing."""

    require_tile_id(tile_id)
    row = connection.execute(
        "SELECT width,height,mask,source,version,updated_at "
        "FROM tidal_connectivity_masks WHERE tile_id=?",
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    width, height = int(row[0]), int(row[1])
    values = np.frombuffer(zlib.decompress(row[2]), dtype=np.uint8)
    if values.size != width * height or not np.all(
        (values == 0) | (values == 1)
    ):
        raise ValueError(f"invalid tidal connectivity snapshot for {tile_id}")
    return {
        "tile_id": tile_id,
        "mask": values.reshape((height, width)).astype(bool),
        "source": row[3],
        "version": int(row[4]),
        "updated_at": row[5],
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _mask_rows_at_depth(
    connection: sqlite3.Connection,
    table: str,
    depth: int,
) -> dict[tuple[int, int], tuple[str, np.ndarray, bytes | None]]:
    """Read exact masks and canonical DEM bytes for one quadtree depth."""

    if table not in {"coastline_masks", "hydrography_masks"}:
        raise ValueError(f"unsupported terrain mask table: {table}")
    rows = connection.execute(
        f"SELECT m.tile_id,t.col,t.row,m.width,m.height,m.mask,t.heightmap "
        f"FROM {table} m JOIN tiles t ON t.tile_id=m.tile_id "
        "WHERE t.depth=?",
        (depth,),
    ).fetchall()
    result = {}
    for tile_id, column, row, width, height, blob, heightmap_blob in rows:
        width, height = int(width), int(height)
        values = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
        if values.size != width * height:
            raise ValueError(
                f"{table} mask for {tile_id} has {values.size} values; "
                f"expected {width * height}"
            )
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{table} mask for {tile_id} is not boolean")
        result[(int(column), int(row))] = (
            tile_id,
            values.reshape((height, width)).astype(bool),
            heightmap_blob,
        )
    return result


def _build_connected_hydrography(
    connection: sqlite3.Connection,
    depth: int,
) -> dict[str, np.ndarray]:
    """Flood four-neighbour WMS components from trusted same-depth sea seeds.

    Seeds come from overlap/touching with GTK50 coastline masks or from a DEM
    tile whose complete finite surface is at or below 0.5 metres. A low sample
    inside an otherwise elevated tile is deliberately insufficient.
    """

    hydro = _mask_rows_at_depth(connection, "hydrography_masks", depth)
    coast = _mask_rows_at_depth(connection, "coastline_masks", depth)
    structure = np.asarray(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=np.uint8,
    )
    labeled = {}
    component_count = 0
    for address, (tile_id, mask, _) in hydro.items():
        labels, count = cast(
            tuple[np.ndarray, int],
            label(mask, structure=structure),
        )
        labeled[address] = (tile_id, mask, labels, count, component_count)
        component_count += count
    if component_count == 0:
        return {
            tile_id: np.zeros_like(mask)
            for tile_id, mask, _ in hydro.values()
        }

    parent = np.arange(component_count, dtype=np.int32)
    seeded = np.zeros(component_count, dtype=bool)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def component_ids(entry, values: np.ndarray) -> np.ndarray:
        labels, base = entry[2], entry[4]
        local = np.unique(labels[values])
        return base + local[local > 0] - 1

    for (column, row), entry in labeled.items():
        _, mask, labels, _, base = entry
        heightmap_blob = hydro[(column, row)][2]
        if heightmap_blob is not None:
            heightmap = np.frombuffer(
                zlib.decompress(heightmap_blob),
                dtype=np.float32,
            )
            if heightmap.size == mask.size:
                heightmap = heightmap.reshape(mask.shape)
                finite = heightmap[np.isfinite(heightmap)]
                if (
                    finite.size
                    and float(np.max(finite)) <= SEA_SEED_MAX_ELEV_M
                ):
                    for node in component_ids(entry, mask):
                        seeded[int(node)] = True

        same_coast = coast.get((column, row))
        if same_coast is not None and same_coast[1].shape == mask.shape:
            for node in component_ids(entry, mask & same_coast[1]):
                seeded[int(node)] = True

        # Join hydrography only once per shared edge: east and north.
        for dc, dr, own_edge, other_edge in (
            (1, 0, (slice(None), -1), (slice(None), 0)),
            (0, 1, (-1, slice(None)), (0, slice(None))),
        ):
            neighbor = labeled.get((column + dc, row + dr))
            if neighbor is None:
                continue
            touching = mask[own_edge] & neighbor[1][other_edge]
            own_labels = labels[own_edge][touching]
            neighbor_labels = neighbor[2][other_edge][touching]
            for own_label, neighbor_label in zip(own_labels, neighbor_labels):
                union(
                    base + int(own_label) - 1,
                    neighbor[4] + int(neighbor_label) - 1,
                )

        # A coastline mask in a neighbouring tile may meet the WMS render at
        # the exact shared edge even when this tile has no coastline pixels.
        for dc, dr, own_edge, coast_edge in (
            (-1, 0, (slice(None), 0), (slice(None), -1)),
            (1, 0, (slice(None), -1), (slice(None), 0)),
            (0, -1, (0, slice(None)), (-1, slice(None))),
            (0, 1, (-1, slice(None)), (0, slice(None))),
        ):
            neighbor_coast = coast.get((column + dc, row + dr))
            if neighbor_coast is None:
                continue
            coast_mask = neighbor_coast[1]
            if mask[own_edge].shape != coast_mask[coast_edge].shape:
                continue
            touching = mask[own_edge] & coast_mask[coast_edge]
            for own_label in np.unique(labels[own_edge][touching]):
                if own_label > 0:
                    seeded[base + int(own_label) - 1] = True

    seed_roots = {find(int(index)) for index in np.flatnonzero(seeded)}
    result = {}
    for tile_id, mask, labels, count, base in labeled.values():
        accepted = [
            local_label
            for local_label in range(1, count + 1)
            if find(base + local_label - 1) in seed_roots
        ]
        result[tile_id] = (
            np.isin(labels, accepted)
            if accepted
            else np.zeros_like(mask)
        )
    return result


def connected_hydrography_for_tile(
    connection: sqlite3.Connection,
    tile_id: str,
) -> np.ndarray | None:
    """Derive one tile's connected WMS hydrography from current source rows."""

    depth, _, _ = require_tile_id(tile_id)
    row = connection.execute(
        "SELECT depth FROM tiles WHERE tile_id=?",
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    if int(row[0]) != depth:
        raise ValueError(f"stored depth does not match tile ID {tile_id}")
    return _build_connected_hydrography(connection, depth).get(tile_id)


def _response(connection: sqlite3.Connection, tile_id: str) -> dict:
    depth, _, _ = require_tile_id(tile_id)
    hydro_row = connection.execute(
        "SELECT width,height,mask FROM hydrography_masks WHERE tile_id=?",
        (tile_id,),
    ).fetchone()
    if hydro_row is None:
        return {"tileId": tile_id, "found": False, "depth": depth}
    width, height = int(hydro_row[0]), int(hydro_row[1])
    raw = np.frombuffer(zlib.decompress(hydro_row[2]), dtype=np.uint8)
    if raw.size != width * height:
        raise ValueError(
            f"hydrography mask for {tile_id} has {raw.size} values; "
            f"expected {width * height}"
        )
    connected = connected_hydrography_for_tile(connection, tile_id)
    if connected is None:
        return {"tileId": tile_id, "found": False, "depth": depth}
    values = connected.astype(np.uint8)
    hydro_count = int(raw.sum())
    connected_count = int(values.sum())
    return {
        "tileId": tile_id,
        "found": True,
        "depth": depth,
        "source": SOURCE,
        "version": VERSION,
        "seaSeedMaxElevation": SEA_SEED_MAX_ELEV_M,
        "shape": [int(connected.shape[0]), int(connected.shape[1])],
        "hydrographyCount": hydro_count,
        "connectedCount": connected_count,
        "rejectedCount": hydro_count - connected_count,
        "digest": hashlib.sha256(values.tobytes()).hexdigest(),
    }


@visible
def derive_tidal_connectivity(tile_id: str) -> dict:
    """Derive same-depth tidal WMS connectivity without writing the database."""

    return _response(db(), tile_id)
