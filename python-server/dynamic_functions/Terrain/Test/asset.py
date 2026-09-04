"""Offline lifecycle checks for the MCP-owned Terrain asset catalog."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from dynamic_functions.Terrain import viewer_assets
from dynamic_functions.Terrain.Asset import catalog
from dynamic_functions.Terrain.Asset import database as assets


@visible
async def asset_lifecycle() -> dict:
    """Verify start, status, bounded listing, filtering, and stop."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = Path(temporary_directory) / "assets.db"
        with (
            patch.object(assets, "DATABASE_PATH", database_path),
            patch.object(assets.atlantis, "client_log", new=AsyncMock()),
            patch.object(assets, "_update_dashboard", new=AsyncMock()),
        ):
            await assets.stop()
            before = assets.status()
            with patch.object(
                viewer_assets, "_LOCAL_ASSETS_DB", database_path
            ):
                try:
                    viewer_assets.startup_assets()
                except viewer_assets.AssetCatalogUnavailable:
                    missing_catalog_rejected = True
                else:
                    missing_catalog_rejected = False
            started = await assets.start()
            running_ux = assets.ux_status()
            connection = assets.db()
            connection.executemany(
                """
                INSERT INTO assets
                    (id, type, enabled, lat, lon, heading_deg, z, properties, cx, cy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "building-1", "BYGNING", 1, 64.1, -51.7, 0, None,
                        '{"height":12,"groundZ":2,"ring":'
                        '[[0,0,10],[2,0,10],[0,2,10]]}',
                        1, 1,
                    ),
                    (
                        "vehicle-1", "KØRETØJ", 1, 64.2, -51.8, 45, 12,
                        '{"wheels":8,"headlightsOn":true}', None, None,
                    ),
                ],
            )
            connection.executemany(
                "INSERT INTO asset_metadata (key, value) VALUES (?, ?)",
                [
                    ("schema_version", "4"),
                    ("vehicle_asset_type", json.dumps("KØRETØJ")),
                    ("structure_asset_type", json.dumps("STRUKTUR")),
                    (
                        "vehicle_definition",
                        json.dumps({
                            "url": "/models/vehicle.glb",
                            "realLengthM": 7.7,
                            "tireDiameterM": 1.27,
                            "altOffsetM": 0.05,
                        }),
                    ),
                    (
                        "structure_definition",
                        json.dumps({"url": "/models/structure.glb"}),
                    ),
                ],
            )
            connection.commit()
            running = assets.status()
            all_assets = assets.list(limit=1)
            buildings = assets.list(asset_type="BYGNING", enabled=True)
            with patch.object(
                viewer_assets, "_LOCAL_ASSETS_DB", database_path
            ):
                startup_payload = viewer_assets.startup_assets()
                building_payload, building_source = viewer_assets.query_buildings(
                    1, 1, 100, 1, 1
                )
            saved_vehicle = catalog.save_vehicle_state({
                "lat": 64.25,
                "lon": -51.75,
                "headingDeg": 405,
                "z": 13.5,
                "terrainDepth": 12.9,
                "terrainTileId": "12-1-2",
                "reason": "test",
            })
            patched_building = catalog.patch_asset(
                "building-1",
                {"enabled": False, "properties": {"reviewed": True}},
            )
            missing_patch = catalog.patch_asset("missing", {"enabled": False})
            stopped = await assets.stop()
            stopped_ux = assets.ux_status()

        return {
            "missingCatalogRejected": missing_catalog_rejected,
            "initiallyStopped": bool(
                not before["running"] and not before["exists"]
            ),
            "starts": bool(
                started["running"]
                and started["asset_count"] == 0
                and database_path.exists()
                and "ASSET DB" in running_ux
                and 'aria-label="on"' in running_ux
            ),
            "status": bool(
                running["running"]
                and running["asset_count"] == 2
                and running["enabled_count"] == 2
                and running["metadata_count"] == 5
                and len(running["type_counts"]) == 2
            ),
            "boundedList": bool(
                all_assets["count"] == 1
                and all_assets["total"] == 2
                and all_assets["limit"] == 1
            ),
            "filteredList": bool(
                buildings["count"] == 1
                and buildings["total"] == 1
                and buildings["assets"][0]["id"] == "building-1"
                and buildings["assets"][0]["properties"]["height"] == 12
            ),
            "strictViewerRead": bool(
                startup_payload["catalogPath"] == str(database_path)
                and startup_payload["vehicle_instances"][0]["id"] == "vehicle-1"
                and building_source == "asset_catalog"
                and building_payload[0]["id"] == "building-1"
            ),
            "vehicleWrite": bool(
                saved_vehicle["vehicleId"] == "vehicle-1"
                and saved_vehicle["state"]["headingDeg"] == 45
                and saved_vehicle["state"]["z"] == 13.5
                and saved_vehicle["state"]["terrainDepth"] == 12
                and saved_vehicle["state"]["terrainTileId"] == "12-1-2"
            ),
            "assetPatch": bool(
                patched_building is not None
                and not patched_building["enabled"]
                and patched_building["properties"]["height"] == 12
                and patched_building["properties"]["reviewed"] is True
                and missing_patch is None
            ),
            "stops": bool(
                not stopped["running"] and stopped["exists"]
                and "ASSET DB" in stopped_ux
                and 'aria-label="off"' in stopped_ux
            ),
        }
