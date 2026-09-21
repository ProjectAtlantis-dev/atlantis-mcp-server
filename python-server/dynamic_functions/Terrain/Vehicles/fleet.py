"""Read-only fleet views for Lobster and other MCP consumers."""
from urllib.parse import quote
from atlantis_economy.gateway import account_for, bank_request
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.host import simulation_host


def _fleet_snapshot():
    principal = current_principal()
    account = account_for(principal)
    assets = bank_request('GET', f"/accounts/{account['id']}/portfolio")['ownedAssets']
    snapshot = simulation_host.command('GET', f'/games/{quote(principal.scenario, safe="")}/snapshot')
    controlled = {v['id']: v for v in snapshot.get('controlledVehicles', [])}
    rows = []
    for asset in assets:
        metadata = asset.get('metadata', {})
        if (metadata.get('world') != principal.scenario or asset.get('status') != 'active'
                or (asset['kind'] != 'vehicle' and not (asset['kind'] == 'structure'
                    and controlled.get(asset['id'], {}).get('presentation') == 'infrastructure'))):
            continue
        vehicle = controlled.get(asset['id'])
        mission = vehicle.get('mission') if vehicle else None
        rows.append({
            'uuid': asset['id'], 'vehicle': metadata.get('terrainAssetId', asset['assetType']),
            'model': asset['assetType'],
            'state': mission['status'] if mission else (vehicle['controlStatus'] if vehicle else 'not-attached'),
            'leg': mission.get('leg', 'outbound') if mission else None,
            'remainingM': mission['remainingM'] if mission else None,
            'phase': vehicle.get('flightPhase') if vehicle else None,
            'missionId': mission['id'] if mission else None,
            'journeyId': mission.get('journeyId', mission['id']) if mission else None,
            'reason': mission.get('reason') if mission else None,
            'attached': vehicle is not None,
        })
    return {'owner': principal.caller, 'world': principal.scenario, 'tick': snapshot['tick'],
            'vehicles': sorted(rows, key=lambda row: row['vehicle'])}


@visible
def fleet() -> dict:
    """Read all your bank-owned vehicles in this world, including unattached ones. Reports UUID, current mission state/leg/distance/reason. One snapshot per call; use fleet_table for terminal display."""
    return _fleet_snapshot()


@visible
def fleet_table() -> str:
    """Display your current vehicle/drone fleet as a terminal table. Run again to refresh. not-attached means no authoritative controller state, not idle or mission success."""
    data = _fleet_snapshot()
    headers = ['Vehicle', 'State', 'Leg', 'Remaining', 'Phase', 'UUID']
    rows = [[v['vehicle'], v['state'], v['leg'] or '-',
             '-' if v['remainingM'] is None else f"{v['remainingM']:.1f} m",
             v['phase'] or '-', v['uuid']] for v in data['vehicles']]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(len(headers))]
    def line(row):
        return ' | '.join(value.ljust(width) for value, width in zip(row, widths))
    result = [f"Fleet · {data['owner']} · {data['world']} · tick {data['tick']}",
              line(headers), '-+-'.join('-' * width for width in widths)]
    result.extend(line(row) for row in rows)
    if not rows:
        result.append('No owned vehicles in this world.')
    result.extend(f"{v['vehicle']}: {v['reason']} · mission {v['missionId']}"
                  for v in data['vehicles'] if v['reason'])
    result.append('Snapshot only. Run fleet_table again to refresh; fleet returns structured rows.')
    return '\n'.join(result)
