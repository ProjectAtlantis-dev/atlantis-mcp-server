"""Folder entry point for Terrain asset catalog tools."""

import atlantis


@index
@visible
async def index() -> None:
    """Open the Terrain asset catalog tools."""
    await atlantis.client_log("Terrain asset catalog tools opened")


@visible
def ground_buildings(apply: bool = False) -> dict:
    """Reconcile existing town building bases to the finest stored EGM2008 DEM. Preview by default; apply persists z/groundZ with source provenance and preserves originalGroundZ. Unknown samples are explicitly reported and unchanged. Does not move footprints or issue new object identities."""
    import sqlite3
    from atlantis_host_adapters.identity import current_principal
    from atlantis_simulation.terrain_adapter import configuration, read_only
    from dynamic_functions.Terrain.Placement.gateway import authorize
    from dynamic_functions.Terrain.Asset.grounding import plan_building_grounding, apply_building_grounding
    principal = current_principal('simulation')
    authorize(principal)
    config = configuration(principal.scenario)
    terrain = read_only(config['terrainDatabase'])
    assets = sqlite3.connect(config['assetDatabase']) if apply else read_only(config['assetDatabase'])
    try:
        updates, unresolved = plan_building_grounding(assets, terrain)
        count = apply_building_grounding(assets, updates) if apply else 0
        return {'applied': count, 'planned': len(updates), 'unresolved': unresolved,
                'maximumChangeM': max((abs(v['groundZ']-v['previousGroundZ']) for v in updates),default=0),
                'verticalDatum': 'EGM2008', 'coordinatesPreserved': True}
    finally:
        assets.close()
        terrain.close()
