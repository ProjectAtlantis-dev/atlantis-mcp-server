"""Directly callable, rollback-only texture persistence check."""

import hashlib
import sqlite3
from pathlib import Path

from dynamic_functions.Terrain.dataforsyningen import _split_metatile
from dynamic_functions.Terrain.Database.database import db
from dynamic_functions.Terrain.Database.textures import (
    TextureClobberError,
    read_texture_payload,
    write_texture_metatile,
)


_FIXTURE_PATH = (
    Path(__file__).with_name("fixtures") / "dataforsyningen_metatile.png"
)
_SOURCE = "dataforsyningen"


def _rows_for(connection, tile_ids: list[str]) -> int:
    marks = ",".join("?" for _ in tile_ids)
    return connection.execute(
        f"SELECT COUNT(*) FROM textures WHERE tile_id IN ({marks})",
        tile_ids,
    ).fetchone()[0]


@visible
def texture_persistence(tile_id: str) -> dict:
    """Verify exact, atomic, idempotent, and no-clobber sibling storage."""

    connection = db()
    fixture = _FIXTURE_PATH.read_bytes()
    children = _split_metatile(fixture, tile_id)
    child_ids = sorted(children)
    connection.execute("SAVEPOINT texture_persistence_test")
    try:
        first_write = write_texture_metatile(
            connection,
            children,
            _SOURCE,
            commit=False,
        )
        duplicate_write = write_texture_metatile(
            connection,
            children,
            _SOURCE,
            commit=False,
        )

        stored = {
            child_id: read_texture_payload(connection, child_id)
            for child_id in child_ids
        }
        exact_round_trip = all(
            stored[child_id] is not None
            and stored[child_id]["texture"] == children[child_id]
            for child_id in child_ids
        )

        changed = dict(children)
        changed_id = child_ids[0]
        changed[changed_id] = children[changed_id] + b"changed"
        clobber_blocked = False
        try:
            write_texture_metatile(
                connection,
                changed,
                _SOURCE,
                commit=False,
            )
        except TextureClobberError:
            clobber_blocked = True
        existing_preserved = all(
            read_texture_payload(connection, child_id)["texture"]
            == children[child_id]
            for child_id in child_ids
        )

        partial_rejected = False
        try:
            write_texture_metatile(
                connection,
                dict(list(children.items())[:-1]),
                _SOURCE,
                commit=False,
            )
        except ValueError:
            partial_rejected = True

        failed_acquisition_rejected = False
        failed_children = dict(children)
        failed_children[child_ids[-1]] = None
        try:
            write_texture_metatile(
                connection,
                failed_children,
                _SOURCE,
                commit=False,
            )
        except TypeError:
            failed_acquisition_rejected = True
        failed_acquisition_preserved = all(
            read_texture_payload(connection, child_id)["texture"]
            == children[child_id]
            for child_id in child_ids
        )

        # Prove a database failure cannot expose an incomplete sibling set.
        failing_children = _split_metatile(fixture, "10-332-212")
        failing_ids = sorted(failing_children)
        connection.execute("DROP TRIGGER IF EXISTS texture_atomicity_test")
        connection.execute(
            "CREATE TEMP TRIGGER texture_atomicity_test "
            "BEFORE INSERT ON textures "
            f"WHEN NEW.tile_id = '{failing_ids[8]}' "
            "BEGIN SELECT RAISE(ABORT, 'injected texture failure'); END"
        )
        atomic_failure_raised = False
        try:
            write_texture_metatile(
                connection,
                failing_children,
                _SOURCE,
                commit=False,
            )
        except sqlite3.IntegrityError:
            atomic_failure_raised = True
        finally:
            connection.execute("DROP TRIGGER IF EXISTS texture_atomicity_test")
        atomic_failure_left_no_siblings = _rows_for(connection, failing_ids) == 0

        combined_digest = hashlib.sha256()
        for child_id in child_ids:
            combined_digest.update(child_id.encode("ascii"))
            combined_digest.update(children[child_id])
        return {
            "tileId": tile_id,
            "childCount": len(child_ids),
            "firstWrite": first_write,
            "duplicateWrite": duplicate_write,
            "exactRoundTrip": exact_round_trip,
            "clobberBlocked": clobber_blocked,
            "existingPreserved": existing_preserved,
            "partialRejected": partial_rejected,
            "failedAcquisitionRejected": failed_acquisition_rejected,
            "failedAcquisitionPreserved": failed_acquisition_preserved,
            "atomicFailureRaised": atomic_failure_raised,
            "atomicFailureLeftNoSiblings": atomic_failure_left_no_siblings,
            "source": _SOURCE,
            "combinedEncodedDigest": combined_digest.hexdigest(),
        }
    finally:
        connection.execute("ROLLBACK TO texture_persistence_test")
        connection.execute("RELEASE texture_persistence_test")
