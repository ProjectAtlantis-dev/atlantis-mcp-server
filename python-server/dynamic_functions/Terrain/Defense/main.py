"""Commands and observations for the fictional, server-owned defense demo."""
from pathlib import Path
from urllib.parse import quote
from atlantis_host_adapters.identity import current_principal
from atlantis_simulation.host import simulation_host


def _request(method, action, payload=None):
    principal = current_principal('simulation')
    return simulation_host.command(method, f'/games/{quote(principal.scenario, safe="")}/{action}', payload) if payload is not None else simulation_host.command(method, f'/games/{quote(principal.scenario, safe="")}/{action}')


@visible
def index() -> dict:
    """Fictional defense demo. Read instructions, set function mode, observe tracks, explicitly call intercept, then observe events/outcome. No real-world sensor or weapon control."""
    return {'module': 'Terrain/Defense', 'start': 'instructions',
            'workflow': ['set_mode', 'spawn_incoming', 'alerts', 'observe', 'intercept', 'events'],
            'scope': 'existing fictional simulation entities only'}


@visible
def instructions() -> str:
    """Read the complete AI/Lobster defense-demo procedure, tool parameters, state meanings, retry rules and limitations."""
    return Path(__file__).with_name('README.md').read_text()


@visible
def set_mode(mode: str = 'functions') -> dict:
    """Choose functions (sensors track; only explicit intercept calls authorize engagement) or automatic (existing game rules authorize). Preserves fleet, structures and live engagements; clears pending unlaunched orders when switching to functions. No credits."""
    if mode not in ('functions', 'automatic'):
        raise ValueError('mode must be functions or automatic')
    return _request('POST', 'defense-mode', {'mode': mode})


@visible
def observe() -> dict:
    """Read detected fictional tracks, current engagement state, available game-layer IDs and readiness, inventory, and event cursor. Undetected targets are omitted. Eligibility is game state, not a real-world recommendation; no action is dispatched."""
    return _request('GET', 'defense-observation')


@visible
def tracks_table() -> list[dict]:
    """Render detected fictional tracks as a Lobster table, with tracking state and currently runnable game-layer choices. Read only."""
    state = observe()
    return [{'target_id': t['id'], 'kind': t['kind'], 'heading_deg': t['headingDeg'], 'state': t['state'],
             'choices': ', '.join(c['siteId'] + '/' + c['layerId'] for c in t['availableActions'])}
            for t in state['tracks']]


@visible
def intercept(target_id: str, site_id: str, layer_id: str) -> dict:
    """Request one fictional intercept using exact target/site/layer IDs from observe. Requires a current complete track; rechecks eligibility at execution, refuses duplicate active engagements, and never queues an untracked target. accepted means launched, not success; read events/observe for outcome."""
    if any(not isinstance(v, str) or not v.strip() for v in (target_id, site_id, layer_id)):
        raise ValueError('Explicit target_id, site_id and layer_id from observe are required')
    return _request('POST', 'intercept', {'targetId': target_id, 'siteId': site_id,
                                       'layerId': layer_id, 'requireTracked': True})


@visible
def events(after_sequence: int = 0) -> dict:
    """Read simulation events after a cursor; retain the greatest sequence returned. Track observation is polled, not an automatic AI callback. Events report launch, miss, simulated interception and target arrival; never infer success from command acceptance."""
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError('after_sequence must be a nonnegative integer')
    return _request('GET', f'events?after={after_sequence}')


@visible
def test_cases() -> list[dict]:
    """List configured fictional layer tests and accepted incoming types. Select site_id/test_layer_id and one incoming_type for spawn_layer_test. Choosing a test fixture never activates its layer."""
    state = observe()
    return [{'site_id': layer['siteId'], 'test_layer_id': layer['layerId'],
             'incoming_types': layer['targetKinds'], 'available': layer['available']}
            for layer in state['layers']]


@visible
def spawn_layer_test(incoming_type: str, request_id: str, site_id: str, test_layer_id: str, heading_deg: float = 90) -> dict:
    """Spawn a synthetic drone, cruise or ballistic label with a chosen direction of travel: heading_deg is clockwise from north, 0 north/90 east/180 south/270 west. Use test_cases for compatible IDs/types. Placement derives from game bounds; motion is a simple 120-second fixture, not realistic flight. Does not activate any layer. Same request_id/terms returns the same target; conflicting reuse rejects. Observe radar detection and explicitly call intercept separately."""
    return _request('POST', 'test-incoming', {'incomingType': incoming_type, 'requestId': request_id,
                                          'siteId': site_id, 'testLayerId': test_layer_id, 'headingDeg': heading_deg})


@visible
def alerts(after_sequence: int = 0) -> dict:
    """Read radar alerts for the AI/Lobster loop. Each alert carries event sequence, incoming type, target ID, detecting sensors, current track state and currently runnable game-layer choices. Poll using next_sequence. Only ready_for_action alerts permit an explicit intercept call; stale/engaged/lost tracks have no runnable choices. Does not activate a layer or start a background AI."""
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError('after_sequence must be a nonnegative integer')
    events_read = events(after_sequence)['events']
    state = observe()
    tracks = {track['id']: track for track in state['tracks']}
    rows = []
    for event in events_read:
        if event['type'] not in ('target-detected', 'target-tracked', 'target-track-lost'):
            continue
        track = tracks.get(event['targetId'])
        actions = track['availableActions'] if track else []
        rows.append({'sequence': event['sequence'], 'event': event['type'], 'target_id': event['targetId'],
                     'incoming_type': event.get('kind'), 'heading_deg': track['headingDeg'] if track else None,
                     'detected_by': event.get('detectedBy', []),
                     'current_state': track['state'] if track else 'not-currently-tracked',
                     'ready_for_action': bool(track and track['state'] == 'tracked' and actions),
                     'available_actions': actions})
    # Advance only through events actually read. A later snapshot may include newer events.
    return {'next_sequence': max([after_sequence] + [event['sequence'] for event in events_read]),
            'observed_tick': state['tick'], 'alerts': rows}


@visible
def spawn_incoming(incoming_type: str, request_id: str, latitude: float, longitude: float,
                   heading_deg: float = 90, approach_distance_m: float = 5000,
                   altitude_m: float = 300, speed_mps: float = 70) -> dict:
    """Send a synthetic incoming toward picked destination coordinates. Types: drone/cruise/ballistic. Heading is travel direction clockwise from north (270 travels west). Start is approach_distance_m behind the destination, altitude_m above its verified terrain height. Simple synthetic motion, not real weapon performance. Map/object picking uses current coordinates; it does not follow a moving object. No defense layer is selected or activated. Observe radar alerts, then call intercept explicitly. Reuse request_id only with identical terms."""
    from dynamic_functions.Terrain.Defense import gateway
    return gateway.spawn(current_principal('simulation'),incoming_type=incoming_type,request_id=request_id,
                         latitude=latitude,longitude=longitude,heading_deg=heading_deg,
                         approach_distance_m=approach_distance_m,altitude_m=altitude_m,speed_mps=speed_mps)


@visible
def asset_status(asset_id: str) -> dict:
    """Read an owned defense component's site, role, sensors, layers and currently runnable synthetic targets. The bank UUID distinguishes copies. Standalone display models reject."""
    from dynamic_functions.Terrain.Defense import gateway
    return gateway.asset_status(current_principal('simulation'), asset_id)


@visible
def intercept_asset(asset_id: str, target_id: str) -> dict:
    """Run the existing fictional interception through this specific bank-owned launcher. Read asset_status first. Requires a currently tracked eligible synthetic target; rechecks the site/layer on invocation. Acceptance means launched, not success."""
    from dynamic_functions.Terrain.Defense import gateway
    return gateway.intercept_asset(current_principal('simulation'), asset_id, target_id)
