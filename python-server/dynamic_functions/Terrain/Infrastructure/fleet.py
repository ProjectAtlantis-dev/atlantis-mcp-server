"""Owner-filtered infrastructure inventory for MCP and terminal inspection."""
from atlantis_economy.gateway import account_for, bank_request
from atlantis_simulation import infrastructure_control as infrastructure
from atlantis_simulation.host import simulation_host


def _inventory():
    principal = infrastructure.principal_for()
    account = account_for(principal)
    assets = bank_request('GET', f"/accounts/{account['id']}/portfolio")['ownedAssets']
    snapshot = simulation_host.command('GET', infrastructure.path(principal, 'snapshot'))
    entities = {entity['id']: entity for entity in snapshot['infrastructure']}
    rows = []
    for asset in assets:
        metadata = asset.get('metadata', {})
        if (asset['kind'] != 'structure' or asset['status'] != 'active'
                or metadata.get('world') != principal.scenario
                or metadata.get('stateAuthority') != 'arctic-simulation'):
            continue
        entity = entities.get(asset['id'])
        rows.append({'uuid': asset['id'], 'model': asset['assetType'],
                     'placed': entity is not None,
                     'siteId': entity.get('siteId') if entity else None,
                     'operationalState': entity.get('operationalState') if entity else None,
                     'equipmentState': entity.get('equipmentState') if entity else None,
                     'movement': entity.get('movement') if entity else None,
                     'scope': entity.get('simulationScope', 'visual-only') if entity else 'not-placed',
                     'position': entity['position'] if entity else None,
                     'componentState': entity.get('componentState') if entity else None})
    return {'owner': principal.caller, 'world': principal.scenario, 'tick': snapshot['tick'],
            'structures': sorted(rows, key=lambda row: (row['model'], row['uuid']))}


@visible
def fleet() -> dict:
    """Read your bank-owned infrastructure and habitat UUIDs, placements, simulation scope, door positions and targets."""
    return _inventory()


@visible
def fleet_table() -> str:
    """Show only your infrastructure/habitat in a terminal table. Door positions are observed state; targets are requests."""
    data = _inventory()
    lines = [f"Infrastructure · {data['owner']} · {data['world']} · tick {data['tick']}",
             'Model | Scope | Outer / target | Inner / target | Freight / target | UUID']
    for row in data['structures']:
        state = row['componentState']
        entry = state.get('entry', state) if state else None
        def door(key):
            return '-' if entry is None else f"{entry[key]:.2f} / {entry['target'][key]:.2f}"
        freight = f"{state['freight']:.2f} / {state['freightTarget']:.2f}" if state and 'freight' in state else '-'
        lines.append(' | '.join([row['model'], row['scope'], door('outer'), door('inner'), freight, row['uuid']]))
    if not data['structures']:
        lines.append('No owned infrastructure in this world.')
    return '\n'.join(lines)
