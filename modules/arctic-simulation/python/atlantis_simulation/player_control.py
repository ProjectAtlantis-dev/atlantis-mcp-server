"""Authenticated player gateway. Identity comes from the host, never tool arguments."""
import json
import os
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import UUID
from atlantis_host_adapters.identity import current_principal
from .host import simulation_host


def player_id(principal):
    if 'simulation' not in principal.permissions:
        raise PermissionError('simulation permission required')
    base = os.environ.get('GREENLAND_GAME_URL', '').rstrip('/')
    token = os.environ.get('GAME_SERVER_AUTHORITY_TOKEN', '')
    if not base or not token:
        raise RuntimeError('Explicit world service and authority token required')
    # Read the existing registry record; no duplicate player or automatic account creation.
    path = '/players/by-external/' + quote(principal.external_user_id, safe='')
    request = Request(base + path, headers={'Authorization': 'Bearer ' + token})
    with urlopen(request, timeout=5) as response:
        result = json.load(response)
    player = result['player']
    if player['externalUserId'] != principal.external_user_id:
        raise PermissionError('Player registry identity mismatch')
    return str(UUID(player['id']))


def execute(principal, operation, *, actor, parameters=None):
    allowed = {'attach': set(), 'claim': set(), 'observe': set(), 'action': set(),
               'move': {'leaseId', 'sequence', 'east', 'north', 'durationMs'},
               'release': {'leaseId'}, 'requestEntry': {'leaseId', 'airlockId', 'expectedRevision'}}
    parameters = parameters or {}
    if operation not in allowed or not isinstance(parameters, dict) or set(parameters) - allowed[operation]:
        raise ValueError('Unsupported player operation or fields')
    payload = dict(parameters, operation=operation, id=player_id(principal), actor=actor)
    return simulation_host.command('POST', f'/games/{quote(principal.scenario, safe="")}/player-control', payload)


def mcp_command(operation, parameters=None):
    principal = current_principal()
    return execute(principal, operation, actor=f'mcp:{principal.external_user_id}:{principal.user_game_id}', parameters=parameters)
