"""Offline regression gate for coarse coverage before fine camera demand."""

from unittest.mock import patch

from dynamic_functions.Terrain import demand


@visible
def camera_prefill_offline() -> dict:
    targets = ["11-922-147", "11-923-147", "11-930-147", "7-57-9"]
    selection = {
        "tileIds": targets,
        "missing": [
            {"tileId": tile_id, "state": "missing", "fallbackTileId": None}
            for tile_id in targets
        ],
    }
    seeds = demand.camera_prefill_ids(selection, targets)
    ready = set()

    def present(_connection, table, ids):
        return set(ids) & ready if table == "dem" else set()

    class Coordinator:
        def refresh(self, candidates):
            self.candidates = candidates
            return {}

        def submit(self, candidates, **kwargs):
            self.candidates = candidates
            return {}

        def status(self):
            return {}

    lanes = Coordinator()
    with patch.object(demand, "_present_ids", side_effect=present), patch.object(
        demand, "eligible_fjord_jobs", return_value=[]
    ):
        first = demand.submit_camera_demand_from_selection(None, selection, lanes)
        ready.add("8-115-18")
        selection["missing"][0].update(state="fallback", fallbackTileId="8-115-18")
        selection["missing"][1].update(state="fallback", fallbackTileId="8-115-18")
        second = demand.submit_camera_demand_from_selection(None, selection, lanes)
        auxiliary = demand.submit_camera_demand_from_selection(
            None, selection, lanes, demand_origin="bathymetry"
        )
        selection["missing"] = [
            {"tileId": targets[0], "state": "fallback", "fallbackTileId": "10-461-73"}
        ]
        covered_seeds = demand.camera_prefill_ids(selection, targets[:3])

    return {
        "seedsDeduplicatedInCameraOrder": seeds == ["8-115-18", "8-116-18", "7-57-9"],
        "coarseDemBeforeRefinement": first["candidates"]["dem"] == [
            "8-115-18", "8-116-18", "7-57-9", *targets[:3]
        ],
        "coarseTextureBeforeRefinement": first["candidates"]["texture"][:2]
        == ["8-112-16", "8-116-16"],
        "readySeedNotReacquired": "8-115-18" not in second["candidates"]["dem"],
        "seedTextureRemainsPrioritized": second["candidates"]["texture"][0] == "8-112-16",
        "seedWaterDependenciesScheduled": "8-115-18" in second["candidates"]["coastline"],
        "coveredRegionNeedsNoSeed": covered_seeds == [],
        "bathymetryDoesNotSeedCameraCoverage": auxiliary["candidates"]["dem"] == targets,
    }


@visible
def camera_coverage_priority_offline() -> dict:
    """Replay the screenshot footprint; normal D8 demand must not trail detail."""

    import sqlite3
    import numpy as np

    from dynamic_functions.Terrain.camera_lod import select_lod_tiles, resolve_lod_coverage
    from dynamic_functions.Terrain.Database import schema
    from dynamic_functions.Terrain.Database.tiles import write_dem
    from dynamic_functions.Terrain.tile_address import require_tile_id

    connection = sqlite3.connect(":memory:")
    try:
        schema.create(connection)
        selection = select_lod_tiles(
            -21779.2055, -3157650.8187, 20000, 13, 316.5867, 13
        )
        selection.update(resolve_lod_coverage(connection, selection))
        targets, coverage = demand.prioritized_selection_ids(selection)
        normal_coarse = {tile_id for tile_id in targets if require_tile_id(tile_id)[0] == 8}
        seeds = demand.camera_prefill_ids(selection, targets)
        candidates = demand.demand_candidates(connection, targets, coverage, seeds)
        queue = candidates["dem"]
        fine_start = next(i for i, tile_id in enumerate(queue) if require_tile_id(tile_id)[0] > 8)
        values = np.full((65, 65), 20, dtype=np.float32)
        for tile_id in queue[:fine_start]:
            write_dem(connection, tile_id, values, "arcticdem_10m", "EGM2008", commit=False)
        coarse_coverage = resolve_lod_coverage(connection, selection)

        # The next poll must keep the already-acquired normal ring's textures
        # ahead of fine texture demand, even though those D8 leaves are exact.
        selection.update(coarse_coverage)
        seeds_after = demand.camera_prefill_ids(selection, targets)
        after = demand.demand_candidates(
            connection, targets, selection["coverageTileIds"], seeds_after
        )
        first_fine_texture = next(
            i for i, tile_id in enumerate(after["texture"])
            if require_tile_id(tile_id)[0] > 8
        )
        normal_metatiles = {demand._metatile_id(tile_id) for tile_id in normal_coarse}
        write_dem(connection, "11-922-147", values, "arcticdem_10m", "EGM2008", commit=False)
        refined = resolve_lod_coverage(connection, selection)
        return {
            "normalDemandAlreadyIncludesCoarse": bool(normal_coarse),
            "distanceOnlyOrderingDelaysCoarse": targets.index(next(iter(normal_coarse))) > 0,
            "normalCoarseRunsBeforeAnyFineTile": normal_coarse <= set(queue[:fine_start]),
            "coarsePassCoversEntireFootprint": all(
                tile["fallbackTileId"] is not None for tile in coarse_coverage["missing"]
            ),
            "readyOuterRingTexturesStayFirst": normal_metatiles <= set(after["texture"][:first_fine_texture]),
            "refinementKeepsNeighborCoverage": all(
                tile["fallbackTileId"] is not None for tile in refined["missing"]
            ) and "11-922-147" in refined["coverageTileIds"],
        }
    finally:
        connection.close()
