"""Owner-only economy authoring. Never exposed as player production claims.

The bank is the UUID/ledger authority. These explicit operator tools set scenario
rules and initial inventory; ongoing production must come from simulation events.
"""
from atlantis_economy.gateway import account_for, bank_request
from atlantis_host_adapters.identity import current_principal


def _request(path, body):
    import atlantis
    principal = current_principal("bank")
    if principal.caller not in atlantis.get_owner_usernames():
        raise PermissionError("Authenticated host owner required for economy authoring")
    return bank_request("POST", path, body)


@visible
def index() -> dict:
    """Scenario-authoring tools for bank accounts, resources, land and production rules."""
    return {"module": "Terrain/Economy/Authority", "visibility": "owner-only"}


@visible
def create_account(display_name: str, account_type: str) -> dict:
    """Create a treasury, warehouse or system account UUID; does not create player identities."""
    if account_type not in {"treasury", "warehouse", "system"}:
        raise ValueError("account_type must be treasury, warehouse or system")
    return _request("/accounts", {"displayName": display_name, "accountType": account_type})


@visible
def register_catalog_vehicle(terrain_asset_id: str) -> dict:
    """Issue a persistent UUID for an existing vehicle to the authenticated host owner. Supports aircraft and boats; does not attach a controller or move the asset. Retries preserve identity."""
    import atlantis
    from atlantis_simulation.terrain_adapter import catalog, configuration

    principal = current_principal("bank")
    if principal.caller not in atlantis.get_owner_usernames():
        raise PermissionError("Authenticated host owner required for vehicle issuance")
    config = configuration(principal.scenario)
    instance = next((item for item in catalog(config)["vehicle_instances"]
                     if item["id"] == terrain_asset_id), None)
    if instance is None:
        raise ValueError("Enabled vehicle instance not found in this world's Terrain catalog")
    account = account_for(principal)
    asset = bank_request("POST", "/assets/issue", {
        "kind": "vehicle", "assetType": instance["definitionId"],
        "ownerAccountId": account["id"], "actorAccountId": account["id"],
        "sourceNamespace": "terrain:" + principal.scenario, "sourceId": terrain_asset_id,
        "metadata": {"world": principal.scenario, "terrainAssetId": terrain_asset_id,
                     "initialCatalogPose": instance}})
    if asset.get("metadata", {}).get("world") != principal.scenario:
        raise ValueError("Existing bank registration has a conflicting world")
    return {"asset": asset, "ownerUsername": principal.caller,
            "terrainAssetId": terrain_asset_id, "world": principal.scenario}


@visible
def register_land(tile_id: str, owner_account_id: str, sale_status: str = "held") -> dict:
    """Register a depth-12 parcel with a bank-issued land-title UUID. Does not sell or charge for it."""
    # Verify the terrain record exists; never mint titles for arbitrary tile strings.
    import os
    import sqlite3
    from pathlib import Path
    filename = os.environ.get("TERRAIN_DB_PATH")
    if not filename:
        raise RuntimeError("TERRAIN_DB_PATH required")
    with sqlite3.connect(Path(filename).resolve(strict=True).as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT depth FROM tiles WHERE tile_id=?", (tile_id,)).fetchone()
    if row is None or row[0] != 12:
        raise ValueError("Parcel must exist at depth 12 in this Terrain database")
    return _request("/parcels/register", {"tileId": tile_id, "ownerAccountId": owner_account_id,
                    "actorAccountId": owner_account_id, "saleStatus": sale_status})


@visible
def issue_resource(asset_type: str, owner_account_id: str, quantity: float,
                   unit: str, origin_tile_id: str) -> dict:
    """Operator-only initial resource issuance; bank requires a registered origin parcel and tracks lot provenance."""
    return _request("/assets/issue", {"kind": "resource_lot", "assetType": asset_type,
                    "ownerAccountId": owner_account_id, "actorAccountId": owner_account_id,
                    "quantity": quantity, "unit": unit, "originTileId": origin_tile_id,
                    "metadata": {"issuanceReason": "operator-scenario-initialization"}})


@visible
def register_production_rule(producer_asset_type: str, output_asset_type: str,
                             output_unit: str, quantity_per_hour: float) -> dict:
    """Authorize a producer type and rate. This does not manufacture resources or advance the tick."""
    return _request("/production/rules", {"producerAssetType": producer_asset_type,
                    "outputAssetType": output_asset_type, "outputUnit": output_unit,
                    "quantityPerHour": quantity_per_hour})


@visible
def register_recipe(recipe_name: str, processor_asset_type: str,
                    inputs: list[dict], outputs: list[dict]) -> dict:
    """Register processing inputs/outputs; actual transformations require a registered game-authority service."""
    return _request("/recipes", {"recipeName": recipe_name, "processorAssetType": processor_asset_type,
                    "inputs": inputs, "outputs": outputs})
