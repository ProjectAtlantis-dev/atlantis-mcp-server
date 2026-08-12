"""Atomic source-texture storage primitives for the Terrain service port.

Connection lifecycle belongs exclusively to ``Database/database.py``. Every
function here accepts that shared connection explicitly and never opens or
closes SQLite itself.
"""

from __future__ import annotations

import datetime
import sqlite3
import uuid
from collections.abc import Mapping

from dynamic_functions.Terrain.tile_address import (
    ancestor_tile_ids,
    require_tile_id,
)


class TextureClobberError(RuntimeError):
    """Raised when an imagery write would replace a different source payload."""

    def __init__(
        self,
        tile_id: str,
        existing_source: str,
        incoming_source: str,
        existing_updated_at: str,
    ) -> None:
        self.tile_id = tile_id
        self.existing_source = existing_source
        self.incoming_source = incoming_source
        self.existing_updated_at = existing_updated_at
        super().__init__(
            f"Refusing to clobber texture {tile_id}: "
            f"existing source={existing_source} "
            f"updated_at={existing_updated_at}, "
            f"incoming source={incoming_source}"
        )


def _validated_metatile_children(
    children: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Return canonical bytes for one complete, aligned 4-by-4 sibling set."""

    if not isinstance(children, Mapping):
        raise TypeError("metatile children must be a mapping")
    if len(children) != 16:
        raise ValueError(
            f"expected 16 texture children, got {len(children)}"
        )

    addresses = {}
    normalized = {}
    for tile_id, payload in children.items():
        depth, column, row = require_tile_id(tile_id)
        if tile_id != f"{depth}-{column}-{row}":
            raise ValueError(f"texture tile id is not canonical: {tile_id!r}")
        if depth < 2 or column >= 1 << depth or row >= 1 << depth:
            raise ValueError(
                f"terrain tile address is outside depth {depth}: {tile_id!r}"
            )
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"texture payload for {tile_id} must be bytes")
        payload = bytes(payload)
        if not payload:
            raise ValueError(f"texture payload for {tile_id} is empty")
        addresses[tile_id] = (depth, column, row)
        normalized[tile_id] = payload

    depths = {address[0] for address in addresses.values()}
    if len(depths) != 1:
        raise ValueError("metatile children must all have the same depth")
    columns = {address[1] for address in addresses.values()}
    rows = {address[2] for address in addresses.values()}
    first_column = min(columns)
    first_row = min(rows)
    if (
        columns != set(range(first_column, first_column + 4))
        or rows != set(range(first_row, first_row + 4))
        or first_column % 4 != 0
        or first_row % 4 != 0
    ):
        raise ValueError(
            "texture children must form one complete aligned 4-by-4 metatile"
        )
    return normalized


def write_texture_metatile(
    db: sqlite3.Connection,
    children: Mapping[str, bytes],
    source: str,
    *,
    commit: bool = True,
) -> bool:
    """Atomically store one complete metatile without replacing source bytes.

    Returns ``True`` when at least one missing child was inserted and ``False``
    when all sixteen source-and-payload pairs were already present. Any
    conflicting child rejects the entire sibling set before insertion.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("texture source must be a non-empty string")
    normalized = _validated_metatile_children(children)
    tile_ids = sorted(normalized)
    marks = ",".join("?" for _ in tile_ids)
    existing_rows = db.execute(
        "SELECT tile_id, source, texture, updated_at FROM textures "
        f"WHERE tile_id IN ({marks})",
        tile_ids,
    ).fetchall()
    existing = {row[0]: row[1:] for row in existing_rows}
    for tile_id in tile_ids:
        row = existing.get(tile_id)
        if row is None:
            continue
        existing_source, existing_payload, existing_updated_at = row
        if existing_source != source or existing_payload != normalized[tile_id]:
            raise TextureClobberError(
                tile_id,
                existing_source,
                source,
                existing_updated_at,
            )

    missing = [tile_id for tile_id in tile_ids if tile_id not in existing]
    if not missing:
        return False

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    values_sql = ",".join("(?, ?, ?, ?)" for _ in missing)
    parameters = []
    for tile_id in missing:
        parameters.extend((tile_id, source, normalized[tile_id], now))

    savepoint = f"texture_metatile_{uuid.uuid4().hex}"
    db.execute(f"SAVEPOINT {savepoint}")
    try:
        # One SQLite statement makes the sibling insert indivisible even when
        # a constraint or trigger rejects a child in the middle of the set.
        db.execute(
            "INSERT INTO textures (tile_id, source, texture, updated_at) "
            f"VALUES {values_sql}",
            parameters,
        )
        db.execute(f"RELEASE {savepoint}")
    except Exception:
        db.execute(f"ROLLBACK TO {savepoint}")
        db.execute(f"RELEASE {savepoint}")
        raise

    if commit:
        db.commit()
    return True


def read_texture_payload(
    db: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read one stored source texture without transforming its exact bytes."""

    require_tile_id(tile_id)
    row = db.execute(
        "SELECT source, texture, updated_at FROM textures WHERE tile_id = ?",
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "tile_id": tile_id,
        "source": row[0],
        "texture": row[1],
        "updated_at": row[2],
    }


def read_texture_with_ancestor(
    db: sqlite3.Connection,
    tile_id: str,
) -> dict | None:
    """Read exact texture data or the nearest stored ancestor without writing."""

    requested_depth, _, _ = require_tile_id(tile_id)
    for candidate_id in ancestor_tile_ids(tile_id, include_self=True):
        payload = read_texture_payload(db, candidate_id)
        if payload is None:
            continue
        resolved_depth, _, _ = require_tile_id(candidate_id)
        return {
            **payload,
            "requested_tile_id": tile_id,
            "resolved_tile_id": candidate_id,
            "depth_delta": requested_depth - resolved_depth,
            "exact": candidate_id == tile_id,
        }
    return None
