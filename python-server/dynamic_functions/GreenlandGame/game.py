from typing import Any, Dict

from ._client import client
from ._identity import authenticated_game_identity


async def _bootstrap() -> Dict[str, Any]:
    identity = authenticated_game_identity()
    result = await client.request(
        "POST",
        "/players/bootstrap",
        body={
            "externalUserId": identity["externalUserId"],
            "displayName": identity["displayName"],
        },
    )
    result["identityAssurance"] = identity["identityAssurance"]
    return result


@visible
async def greenland_game_status() -> Dict[str, Any]:
    """Return authoritative Greenland game-server status and renderer target."""
    return await client.request("GET", "/health", authority=False)


@visible
async def join_greenland_game() -> Dict[str, Any]:
    """Create or return the authenticated player's profile and bank-owned starter boat."""
    return await _bootstrap()


@visible
async def get_my_greenland_state() -> Dict[str, Any]:
    """Return the caller's authoritative player, boats, cargo, fuel, and voyage state."""
    identity = authenticated_game_identity()
    await _bootstrap()
    external_id = client.path_value(identity["externalUserId"])
    result = await client.request("GET", f"/players/by-external/{external_id}")
    result["identityAssurance"] = identity["identityAssurance"]
    return result


@visible
async def get_greenland_world(after_event_seq: int = 0) -> Dict[str, Any]:
    """Return ports, explicit routes, markets, boats, active voyages, and new world events."""
    authenticated_game_identity()
    return await client.request("GET", f"/world?afterEventSeq={max(0, after_event_seq)}")


@visible
async def quote_my_boat_voyage(
    vehicle_asset_id: str,
    destination_node_code: str,
) -> Dict[str, Any]:
    """Quote route, fuel, departure, and ETA for one of the caller's docked boats."""
    identity = authenticated_game_identity()
    await _bootstrap()
    return await client.request(
        "POST",
        "/voyages/quote",
        body={
            "externalUserId": identity["externalUserId"],
            "vehicleAssetId": vehicle_asset_id,
            "destinationNodeCode": destination_node_code,
        },
    )


@visible
async def depart_my_boat(
    vehicle_asset_id: str,
    destination_node_code: str,
    idempotency_key: str,
) -> Dict[str, Any]:
    """Begin a server-timed voyage; retries with the same key return the same transit."""
    identity = authenticated_game_identity()
    await _bootstrap()
    return await client.request(
        "POST",
        "/voyages/depart",
        body={
            "externalUserId": identity["externalUserId"],
            "vehicleAssetId": vehicle_asset_id,
            "destinationNodeCode": destination_node_code,
            "idempotencyKey": idempotency_key,
        },
    )


@visible
async def get_boat_voyage(transit_id: str) -> Dict[str, Any]:
    """Return authoritative progress, interpolated WGS84 position, and arrival state."""
    authenticated_game_identity()
    return await client.request("GET", f"/voyages/{client.path_value(transit_id)}")


@visible
async def load_my_resource_lot(
    vehicle_asset_id: str,
    resource_asset_id: str,
) -> Dict[str, Any]:
    """Load a bank-verified resource-lot UUID onto the caller's docked boat."""
    identity = authenticated_game_identity()
    await _bootstrap()
    return await client.request(
        "POST",
        "/cargo/load",
        body={
            "externalUserId": identity["externalUserId"],
            "vehicleAssetId": vehicle_asset_id,
            "resourceAssetId": resource_asset_id,
        },
    )


@visible
async def unload_my_resource_lot(
    vehicle_asset_id: str,
    resource_asset_id: str,
) -> Dict[str, Any]:
    """Unload one bank-verified resource lot from the caller's docked boat."""
    identity = authenticated_game_identity()
    return await client.request(
        "POST",
        "/cargo/unload",
        body={
            "externalUserId": identity["externalUserId"],
            "vehicleAssetId": vehicle_asset_id,
            "resourceAssetId": resource_asset_id,
        },
    )
