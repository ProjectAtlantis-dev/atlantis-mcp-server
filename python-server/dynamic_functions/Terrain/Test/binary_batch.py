"""Deterministic gate for browser-compatible binary-v1 composition."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import struct
import zlib

import numpy as np

from dynamic_functions.Terrain.binary_batch import encode_composed_tiles_binary


_EXPECTED_DIGEST = (
    "09a8408a25c1444ccf879a031a187505d875e3d90e54cfdc5a355f1690fe3498"
)


def _ready_tile(tile_id: str, values: np.ndarray) -> dict:
    payload = values.astype("<f4", copy=False).tobytes()
    return {
        "tileId": tile_id,
        "dem": {
            "state": "ready",
            "exact": True,
            "resolvedTileId": tile_id,
            "heightmap": {
                "state": "ready",
                "shape": [int(value) for value in values.shape],
                "dtype": "float32",
                "digest": hashlib.sha256(payload).hexdigest(),
                "contentBase64": base64.b64encode(payload).decode("ascii"),
            },
        },
        "texture": {
            "state": "ready",
            "digest": "fixture-texture",
            "contentBase64": base64.b64encode(b"jpeg fixture").decode("ascii"),
        },
    }


@visible
def binary_batch_offline() -> dict:
    """Prove alignment, block order, reuse, isolation, and stable bytes."""

    reused_values = np.asarray([[1.25, -2.5], [3.75, 4.5]], dtype="<f4")
    sent_values = np.arange(9, dtype="<f4").reshape((3, 3)) / np.float32(4)
    reused = _ready_tile("2-1-1", reused_values)
    sent = _ready_tile("2-1-2", sent_values)
    corrupt = _ready_tile("2-2-1", np.ones((2, 2), dtype="<f4"))
    corrupt["dem"]["heightmap"]["digest"] = "not-the-payload-digest"
    composition = {
        "tiles": [
            reused,
            sent,
            corrupt,
            {
                "tileId": "2-2-2",
                "dem": {"state": "missing", "heightmap": {"state": "missing"}},
                "texture": {"state": "missing"},
            },
        ],
        "tileCount": 4,
        "readOnly": True,
        "networkAccess": False,
        "scheduledWork": False,
    }
    original = copy.deepcopy(composition)
    reused_payload = reused_values.tobytes()
    reused_digest = f"{zlib.crc32(reused_payload) & 0xFFFFFFFF:08x}"
    sent_payload = sent_values.tobytes()
    sent_digest = f"{zlib.crc32(sent_payload) & 0xFFFFFFFF:08x}"

    body, header = encode_composed_tiles_binary(
        composition,
        {"2-1-1": reused_digest.upper()},
    )
    header_length = struct.unpack_from("<I", body, 0)[0]
    header_bytes = body[4 : 4 + header_length]
    decoded = json.loads(header_bytes)
    samples_offset = 4 + header_length
    sent_from_wire = np.frombuffer(
        body,
        dtype="<f4",
        count=sent_values.size,
        offset=samples_offset,
    ).reshape(sent_values.shape)

    # Independent construction of the browser wire contract.
    oracle_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    oracle_header += b" " * (-(len(oracle_header) + 4) % 4)
    oracle = struct.pack("<I", len(oracle_header)) + oracle_header + sent_payload

    invalid_known_rejected = False
    try:
        encode_composed_tiles_binary(composition, {"2-1-1": "not-a-crc"})
    except ValueError:
        invalid_known_rejected = True

    return {
        "format": decoded["format"],
        "aligned": samples_offset % 4 == 0,
        "browserFields": bool(
            decoded["tiles"][0]["id"] == "2-1-1"
            and decoded["tiles"][0]["heightmap"] == reused_digest
            and decoded["tiles"][0]["resolution"] == 2
        ),
        "knownDigestReused": bool(
            decoded["tiles"][0]["heightmapBytes"] == 0
            and decoded["tilesReused"] == 1
        ),
        "unknownDigestTransferred": bool(
            decoded["tiles"][1]["heightmap"] == sent_digest
            and decoded["tiles"][1]["heightmapBytes"] == len(sent_payload)
            and np.array_equal(sent_from_wire, sent_values)
        ),
        "missingCarriesNoBlock": decoded["tiles"][3]["heightmapBytes"] is None,
        "corruptDomainIsolated": bool(
            decoded["tiles"][2]["dem"]["state"] == "error"
            and decoded["tiles"][2]["heightmapBytes"] is None
            and decoded["tiles"][1]["heightmapBytes"] == len(sent_payload)
        ),
        "embeddedBase64Removed": "contentBase64" not in header_bytes.decode(),
        "compositionUnchanged": composition == original,
        "exactOracleParity": body == oracle,
        "noTrailingBytes": len(body) == samples_offset + len(sent_payload),
        "invalidKnownRejected": invalid_known_rejected,
        "digest": hashlib.sha256(body).hexdigest(),
        "stableDigest": hashlib.sha256(body).hexdigest() == _EXPECTED_DIGEST,
        "contentLength": len(body),
    }
