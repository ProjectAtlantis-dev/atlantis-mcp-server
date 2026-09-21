"""Investor demo controls: every action delegates to an existing state owner."""
from urllib.parse import quote
from atlantis_economy.gateway import account_for, bank_request
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.host import simulation_host


def _snapshot():
    principal=current_principal()
    return principal,simulation_host.command('GET',f'/games/{quote(principal.scenario,safe="")}/snapshot')


@visible
def index() -> dict:
    """Start with briefing, then drive, fly, introduce a test threat and operate the habitat. status reports live completion rather than command acceptance."""
    return {'module':'Terrain/Demo','visibility':'owner-only','sequence':['briefing','Terrain/Vehicles/drive_to','Terrain/Vehicles/fly_to','Terrain/Defense/instructions','defense_threat','Terrain/Defense/observe','Terrain/Defense/intercept','Terrain/Infrastructure/component_command','status']}


@visible
def briefing() -> dict:
    """Explain the investor demo using this owner's actual bank assets and the live simulation."""
    principal,snapshot=_snapshot()
    account=account_for(principal)
    assets=bank_request('GET',f"/accounts/{account['id']}/portfolio")['ownedAssets']
    return {'owner':principal.caller,'world':principal.scenario,
        'assets':[{'uuid':a['id'],'catalogId':a.get('metadata',{}).get('terrainAssetId'),
                   'model':a['assetType'],'owner':a['ownerName'],'kind':a['kind']} for a in assets],
        'sequence':[
          {'step':'Ground logistics','command':'Terrain/Vehicles/drive_to','show':'AMV follows a terrain-validated route; wheel travel comes from the server.'},
          {'step':'Drone flight','command':'Terrain/Vehicles/fly_to','show':'VTOL takeoff, cruise and arrival; position and rotor state come from the server.'},
          {'step':'Layered defense','command':'Terrain/Defense/instructions','show':'Functions mode: simulated detection, AI/Lobster observation, explicit intercept function, server outcome.'},
          {'step':'Habitat access','command':'Terrain/Infrastructure/component_command','show':'Open outer door, close and wait, then open inner door; interlocks and animation follow server positions.'}],
        'defense':{'description':'Fictional layered-defense gameplay simulation; not a model of real weapon performance.',
          'layers':{'upper-tier':'high-altitude ballistic targets','middle-tier':'medium-range mixed targets','point-defense':'nearby drones and cruise targets','directed-energy':'close-range drone engagement'},
          'automatic':snapshot.get('automaticDefense'),
          'distinctEntities':'Defense radar/launcher/logistics entities are separate from bank-owned catalog vehicles.'},
        'limitations':['Ground navigation plans a complete destination route, with local terrain checks and replanning.',
                       'VTOL supports Black Hornet/Osprey; RQ-180 supports flyover/loiter without landing; boats use verified water routes.',
                       'Habitat demo models door/access interlocks; pressure, oxygen and power production are not simulated.']}


@visible
def status() -> dict:
    """Read actual mission progress, flight/rotor state, defense outcomes and habitat door positions."""
    principal,snapshot=_snapshot()
    return {'owner':principal.caller,'world':principal.scenario,'tick':snapshot['tick'],
        'vehicles':[{k:v.get(k) for k in ('id','terrainAssetId','definitionId','authority','lat','lon','position','speedMps','flightPhase','rotorRpm','mission')} for v in snapshot.get('controlledVehicles',[])],
        'defense':{'counts':snapshot.get('counts'),'statistics':snapshot.get('statistics'),
                   'sites':snapshot.get('sites'),'targets':snapshot.get('targets'),'engagements':snapshot.get('engagements')},
        'habitat':[{'id':v['id'],'model':v['modelId'],'components':v.get('componentState')} for v in snapshot.get('infrastructure',[]) if v.get('componentState')]}


@visible
def defense_threat(request_id: str) -> dict:
    """Introduce one fictional drone test target near the first deployed defense site. Server sensor/engagement rules decide the outcome; no success is scripted. A unique request_id becomes its target ID."""
    principal,snapshot=_snapshot()
    if not isinstance(request_id,str) or not request_id.strip() or len(request_id)>100:
        raise ValueError('request_id must contain 1..100 characters')
    sites=[s for s in snapshot['sites'] if s['readiness']=='deployed']
    if not sites:raise ValueError('No deployed defense site; use ArcticSimulation/deploy_site first')
    site=sites[0];p=site['position'];target_id='investor-demo:'+request_id
    existing=next((t for t in snapshot['targets'] if t['id']==target_id),None)
    if existing:return {'target':existing,'alreadyExists':True}
    result=simulation_host.command('POST',f'/games/{quote(principal.scenario,safe="")}/targets',{
        'id':target_id,'catalogId':'shahed','kind':'drone','label':'Demo test drone',
        'start':{'x':p['x']-4500,'y':p['y']+1200,'z':p['z']+300},
        'destination':{'x':p['x'],'y':p['y'],'z':p['z']+25},'speedMps':70})
    return {'accepted':result,'site':site['id'],'observe':'Terrain/Defense/observe and Terrain/Defense/events; functions mode requires an explicit intercept call'}
