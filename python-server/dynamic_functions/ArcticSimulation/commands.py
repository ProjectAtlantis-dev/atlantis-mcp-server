"""Thin strategic tools for the MCP-owned headless simulation service."""

from __future__ import annotations

import json
from urllib.parse import quote

from atlantis_simulation.host import simulation_host
from atlantis_host_adapters.identity import authorized_scenario


def _game_path(game_id: str, action: str) -> str:
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("game_id must be a non-empty string")
    return f"/games/{quote(authorized_scenario(game_id.strip()), safe='')}/{action}"


@visible
def start(host: str = "127.0.0.1", port: int = 5190) -> dict:
    """Start the MCP-supervised authoritative simulation child service."""
    return simulation_host.start(host, port)


@visible
def status() -> dict:
    """Report simulation process health without changing it."""
    return simulation_host.status()


@visible
def stop() -> dict:
    """Stop the authoritative simulation child service."""
    return simulation_host.stop()


@visible
def reset_game(
    game_id: str = "default",
    latitude: float = 64.1814,
    longitude: float = -51.6941,
    automatic_defense: bool = True,
    configuration_json: str = "",
) -> dict:
    """Reset an isolated scenario, not the bank/world. Does not import or alter terrain assets."""
    configuration = json.loads(configuration_json) if configuration_json.strip() else {}
    if not isinstance(configuration, dict):
        raise ValueError("configuration_json must decode to an object")
    configuration["origin"] = {"latitude": latitude, "longitude": longitude, "altitudeM": 0}
    configuration["automaticDefense"] = automatic_defense
    simulation_host.command("POST", _game_path(game_id, "reset"), configuration)
    return simulation_host.command("GET", _game_path(game_id, "snapshot"))


@visible
def spawn_shahed(
    game_id: str = "default",
    target_id: str | None = None,
    start_x: float = -9000,
    start_y: float = 5200,
    altitude_m: float = 750,
    destination_x: float = 0,
    destination_y: float = 0,
    destination_altitude_m: float = 35,
    speed_mps: float = 76,
) -> dict:
    """Spawn one server-owned Shahed target in the game's local ENU frame."""
    payload = {
        "catalogId": "shahed",
        "kind": "drone",
        "label": "OWA drone",
        "start": {"x": start_x, "y": start_y, "z": altitude_m},
        "destination": {"x": destination_x, "y": destination_y, "z": destination_altitude_m},
        "speedMps": speed_mps,
    }
    if target_id is not None:
        payload["id"] = target_id
    return simulation_host.command("POST", _game_path(game_id, "targets"), payload)


@visible
def spawn_ballistic(
    game_id: str = "default",
    target_id: str | None = None,
    start_x: float = -22000,
    start_y: float = 11000,
    start_altitude_m: float = 100,
    destination_x: float = 0,
    destination_y: float = 0,
    duration_seconds: float = 24,
    apex_m: float = 8500,
) -> dict:
    """Spawn a server-owned ballistic test target on a configurable arc."""
    payload = {
        "catalogId": "ballistic",
        "kind": "ballistic",
        "label": "ballistic test target",
        "trajectory": "ballistic",
        "start": {"x": start_x, "y": start_y, "z": start_altitude_m},
        "destination": {"x": destination_x, "y": destination_y, "z": 0},
        "durationSeconds": duration_seconds,
        "apexM": apex_m,
    }
    if target_id is not None:
        payload["id"] = target_id
    return simulation_host.command("POST", _game_path(game_id, "targets"), payload)


@visible
def intercept(game_id: str = "default", target_id: str | None = None, layer_id: str | None = None) -> dict:
    """Authorize an intercept; tracking, launch, flight, and resolution stay in the sim."""
    return simulation_host.command("POST", _game_path(game_id, "intercept"), {
        "targetId": target_id,
        "layerId": layer_id,
    })


@visible
def deploy_site(
    game_id: str = "default",
    site_id: str = "forward-defense",
    layer_ids: str = "point-defense",
    position_x: float = 0,
    position_y: float = 0,
    altitude_m: float = 0,
    build_seconds: float = 15,
    sensor_range_m: float = 15000,
) -> dict:
    """Place a defense site; it becomes operational after server-owned construction."""
    layers = [item.strip() for item in layer_ids.split(",") if item.strip()]
    if not layers:
        raise ValueError("layer_ids must contain at least one layer")
    return simulation_host.command("POST", _game_path(game_id, "sites"), {
        "id": site_id,
        "layerIds": layers,
        "position": {"x": position_x, "y": position_y, "z": altitude_m},
        "buildSeconds": build_seconds,
        "sensorRangeM": sensor_range_m,
    })


@visible
def snapshot(game_id: str = "default") -> dict:
    """Read canonical positions, engagements, readiness, and running counts."""
    return simulation_host.command("GET", _game_path(game_id, "snapshot"))


@visible
def events(game_id: str = "default", after_sequence: int = 0) -> dict:
    """Read the append-only event view after a known sequence number."""
    path = f"{_game_path(game_id, 'events')}?after={int(after_sequence)}"
    return simulation_host.command("GET", path)
