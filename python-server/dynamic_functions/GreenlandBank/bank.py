from typing import Any, Dict, List, Optional

from ._client import client
from ._identity import authenticated_game_identity


async def _my_account() -> Dict[str, Any]:
    identity = authenticated_game_identity()
    account = await client.request(
        "POST",
        "/accounts/resolve",
        body={
            "externalUserId": identity["externalUserId"],
            "displayName": identity["displayName"],
            "accountType": "player",
        },
        authority=True,
    )
    account["identityAssurance"] = identity["identityAssurance"]
    return account


@visible
async def game_bank_status() -> Dict[str, Any]:
    """Return the central Greenland game-bank service status."""
    return await client.request("GET", "/health")


@visible
async def open_game_bank_account() -> Dict[str, Any]:
    """Open or return the caller's separate Greenland game-bank account."""
    account = await _my_account()
    return {"success": True, "account": account}


@visible
async def get_my_game_portfolio() -> Dict[str, Any]:
    """Return land, assets, resource lots, custody, and game-credit balance for the caller."""
    account = await _my_account()
    portfolio = await client.request("GET", f"/accounts/{account['id']}/portfolio")
    balance = await client.request("GET", f"/accounts/{account['id']}/balance")
    return {
        "success": True,
        "identityAssurance": account["identityAssurance"],
        "portfolio": portfolio,
        "balance": balance,
    }


@visible
async def verify_game_asset(asset_id: str) -> Dict[str, Any]:
    """Verify an asset or resource-lot UUID against the authoritative game bank."""
    return await client.request("GET", f"/assets/{asset_id}/verify")


@visible
async def get_game_asset_provenance(asset_id: str) -> Dict[str, Any]:
    """Return issuance, transfers, custody, transformations, and parent-lot history."""
    return await client.request("GET", f"/assets/{asset_id}/provenance")


@visible
async def get_game_land_title(tile_id: str) -> Dict[str, Any]:
    """Return the title UUID and current owner for a native depth-12 terrain tile."""
    return await client.request("GET", f"/parcels/{tile_id}")


@visible
async def purchase_game_land(tile_ids: List[str], idempotency_key: str) -> Dict[str, Any]:
    """Purchase one or more bank-listed native depth-12 terrain tiles for the authenticated player."""
    account = await _my_account()
    return await client.request(
        "POST",
        "/parcels/purchase",
        body={
            "tileIds": tile_ids,
            "buyerAccountId": account["id"],
            "actorAccountId": account["id"],
            "idempotencyKey": idempotency_key,
        },
        authority=True,
    )


@visible
async def settle_my_warehouse_quote(quote_id: str, idempotency_key: str) -> Dict[str, Any]:
    """Settle a full or partial warehouse quote; partial sales return sold and remainder UUID lots."""
    account = await _my_account()
    return await client.request(
        "POST",
        f"/warehouse/quotes/{quote_id}/settle",
        body={
            "buyerAccountId": account["id"],
            "actorAccountId": account["id"],
            "idempotencyKey": idempotency_key,
        },
        authority=True,
    )


@visible
async def transfer_my_game_asset(asset_id: str, to_account_id: str) -> Dict[str, Any]:
    """Transfer one active asset owned by the authenticated caller."""
    account = await _my_account()
    return await client.request(
        "POST",
        f"/assets/{asset_id}/transfer",
        body={
            "fromOwnerAccountId": account["id"],
            "toOwnerAccountId": to_account_id,
            "actorAccountId": account["id"],
        },
        authority=True,
    )


@visible
async def deposit_my_asset_at_warehouse(asset_id: str, warehouse_account_id: str) -> Dict[str, Any]:
    """Authorize a registered warehouse to hold, but not own, one of the caller's assets."""
    account = await _my_account()
    return await client.request(
        "POST",
        f"/assets/{asset_id}/custody",
        body={
            "ownerAccountId": account["id"],
            "toCustodianAccountId": warehouse_account_id,
            "actorAccountId": account["id"],
        },
        authority=True,
    )


@visible
async def list_my_warehouse_asset_for_sale(
    asset_id: str,
    warehouse_account_id: str,
    idempotency_key: str,
    minimum_gross_amount: float = 0,
    expires_in_seconds: int = 86400,
) -> Dict[str, Any]:
    """Authorize a warehouse to quote one deposited UUID asset for a limited period."""
    account = await _my_account()
    return await client.request(
        "POST",
        "/warehouse/listings",
        body={
            "assetId": asset_id,
            "sellerAccountId": account["id"],
            "warehouseAccountId": warehouse_account_id,
            "minimumGrossAmount": minimum_gross_amount,
            "currency": "GLC",
            "expiresInSeconds": expires_in_seconds,
            "idempotencyKey": idempotency_key,
        },
        authority=True,
    )


@visible
async def bank_create_managed_account(
    display_name: str,
    account_type: str,
) -> Dict[str, Any]:
    """Bank-owner tool: create a warehouse, treasury, or system game-bank account."""
    if account_type not in {"warehouse", "treasury", "system"}:
        raise ValueError("account_type must be warehouse, treasury, or system")
    return await client.request(
        "POST",
        "/accounts",
        body={"displayName": display_name, "accountType": account_type},
        authority=True,
    )


@visible
async def bank_register_service_identity(
    service_name: str,
    service_type: str,
    owner_account_id: str,
) -> Dict[str, Any]:
    """Bank-owner tool: issue a one-time credential for a named warehouse or game authority."""
    if service_type not in {"warehouse", "game_authority"}:
        raise ValueError("service_type must be warehouse or game_authority")
    return await client.request(
        "POST",
        "/services/register",
        body={
            "serviceName": service_name,
            "serviceType": service_type,
            "ownerAccountId": owner_account_id,
        },
        authority=True,
    )


@visible
async def bank_transfer_credits(
    from_account_id: str,
    to_account_id: str,
    amount: float,
    idempotency_key: str,
    currency: str = "GLC",
) -> Dict[str, Any]:
    """Bank-owner tool: post a balanced game-credit transfer between bank accounts."""
    return await client.request(
        "POST",
        "/credits/transfer",
        body={
            "fromAccountId": from_account_id,
            "toAccountId": to_account_id,
            "amount": amount,
            "currency": currency,
            "actorAccountId": from_account_id,
            "idempotencyKey": idempotency_key,
        },
        authority=True,
    )


@visible
async def bank_register_parcel(
    tile_id: str,
    treasury_account_id: str,
    sale_status: str = "held",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bank-owner tool: register a depth-12 terrain parcel and mint its title UUID."""
    return await client.request(
        "POST",
        "/parcels/register",
        body={
            "tileId": tile_id,
            "ownerAccountId": treasury_account_id,
            "actorAccountId": treasury_account_id,
            "saleStatus": sale_status,
            "metadata": metadata or {},
        },
        authority=True,
    )


@visible
async def bank_issue_resource_lot(
    asset_type: str,
    owner_account_id: str,
    quantity: float,
    unit: str,
    origin_tile_id: str,
    actor_account_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bank-owner tool: issue a lot only after authoritative game production validation."""
    return await client.request(
        "POST",
        "/assets/issue",
        body={
            "kind": "resource_lot",
            "assetType": asset_type,
            "ownerAccountId": owner_account_id,
            "quantity": quantity,
            "unit": unit,
            "originTileId": origin_tile_id,
            "actorAccountId": actor_account_id,
            "metadata": metadata or {},
        },
        authority=True,
    )


@visible
async def bank_issue_serialized_asset(
    kind: str,
    asset_type: str,
    owner_account_id: str,
    actor_account_id: str,
    origin_tile_id: Optional[str] = None,
    location_tile_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bank-owner tool: mint one UUID for a vehicle, machine, structure, or other unique item."""
    allowed = {"vehicle", "equipment", "machinery", "powerplant", "warehouse", "structure"}
    if kind not in allowed:
        raise ValueError(f"kind must be one of: {', '.join(sorted(allowed))}")
    return await client.request(
        "POST",
        "/assets/issue",
        body={
            "kind": kind,
            "assetType": asset_type,
            "ownerAccountId": owner_account_id,
            "originTileId": origin_tile_id,
            "locationTileId": location_tile_id,
            "actorAccountId": actor_account_id,
            "metadata": metadata or {},
        },
        authority=True,
    )


@visible
async def bank_deploy_asset(
    asset_id: str,
    owner_account_id: str,
    tile_id: str,
    actor_account_id: str,
) -> Dict[str, Any]:
    """Bank-owner tool: record placement of a UUID machine on an owned terrain parcel."""
    return await client.request(
        "POST",
        f"/assets/{asset_id}/deploy",
        body={
            "ownerAccountId": owner_account_id,
            "tileId": tile_id,
            "actorAccountId": actor_account_id,
        },
        authority=True,
    )


@visible
async def bank_register_production_rule(
    producer_asset_type: str,
    output_asset_type: str,
    output_unit: str,
    quantity_per_hour: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bank-owner tool: allow a deployed producer type to mint a resource at a fixed rate."""
    return await client.request(
        "POST",
        "/production/rules",
        body={
            "producerAssetType": producer_asset_type,
            "outputAssetType": output_asset_type,
            "outputUnit": output_unit,
            "quantityPerHour": quantity_per_hour,
            "metadata": metadata or {},
        },
        authority=True,
    )


@visible
async def bank_register_recipe(
    recipe_name: str,
    processor_asset_type: str,
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bank-owner tool: register conserved input lots and resulting output lots."""
    return await client.request(
        "POST",
        "/recipes",
        body={
            "recipeName": recipe_name,
            "processorAssetType": processor_asset_type,
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata or {},
        },
        authority=True,
    )
