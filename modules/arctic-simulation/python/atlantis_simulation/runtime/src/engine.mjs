import { createHash, randomUUID } from 'node:crypto';
import { VehicleLogisticsSystem } from './vehicle-logistics.mjs';
import {movementDenial,vehicleWorldPosition} from './infrastructure-access.mjs';
import { InfrastructureState } from './infrastructure-state.mjs';
import { GroundControls } from './ground-controls.mjs';
import { PlayerPresence, protectedEntrance } from './player-presence.mjs';

const DEFAULT_CONFIG = Object.freeze({
  tickRateHz: 30,
  automaticDefense: true,
  origin: { latitude: 64.1814, longitude: -51.6941, altitudeM: 0 },
  defendedPoint: { x: 0, y: 0, z: 0 },
  sites: [{
    id: 'nuuk-defense',
    position: { x: 900, y: -450, z: 12 },
    sensorRangeM: 32000,
    trackBuildSeconds: 0.35,
    readiness: 'deployed',
    layers: [
      { id: 'upper-tier', targetKinds: ['ballistic'], minRangeM: 4500, maxRangeM: 30000, minAltitudeM: 2200, maxAltitudeM: 18000, interceptorSpeedMps: 1700, fuseRadiusM: 130, reloadSeconds: 4.5, inventory: 8, effectiveness: 0.84 },
      { id: 'middle-tier', targetKinds: ['ballistic', 'cruise', 'drone'], minRangeM: 1800, maxRangeM: 15000, minAltitudeM: 180, maxAltitudeM: 9000, interceptorSpeedMps: 820, fuseRadiusM: 75, reloadSeconds: 2.4, inventory: 16, effectiveness: 0.78 },
      { id: 'point-defense', targetKinds: ['cruise', 'drone'], minRangeM: 300, maxRangeM: 6500, minAltitudeM: 20, maxAltitudeM: 3200, interceptorSpeedMps: 430, fuseRadiusM: 45, reloadSeconds: 0.9, inventory: 32, effectiveness: 0.72 },
      { id: 'directed-energy', kind: 'laser', targetKinds: ['drone'], minRangeM: 80, maxRangeM: 2800, minAltitudeM: 10, maxAltitudeM: 1800, dwellSeconds: 1.4, reloadSeconds: 0.4, inventory: 1, effectiveness: 0.95 },
    ],
  }],
});

function finite(value, fallback, minimum = -Infinity, maximum = Infinity) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.min(maximum, parsed)) : fallback;
}

function point(value, fallback = { x: 0, y: 0, z: 0 }) {
  return {
    x: finite(value?.x, fallback.x),
    y: finite(value?.y, fallback.y),
    z: finite(value?.z, fallback.z),
  };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function direction(from, to) {
  const length = distance(from, to);
  if (length <= 1e-9) return { x: 0, y: 0, z: 0 };
  return { x: (to.x - from.x) / length, y: (to.y - from.y) / length, z: (to.z - from.z) / length };
}

function moveToward(position, destination, distanceM) {
  const remaining = distance(position, destination);
  if (remaining <= distanceM) return { position: { ...destination }, arrived: true };
  const unit = direction(position, destination);
  return {
    position: {
      x: position.x + unit.x * distanceM,
      y: position.y + unit.y * distanceM,
      z: position.z + unit.z * distanceM,
    },
    arrived: false,
  };
}

function normalizedConfig(input = {}) {
  const source = input && typeof input === 'object' ? input : {};
  const sites = Array.isArray(source.sites) && source.sites.length > 0 ? source.sites : DEFAULT_CONFIG.sites;
  return {
    tickRateHz: finite(source.tickRateHz, DEFAULT_CONFIG.tickRateHz, 1, 120),
    automaticDefense: source.automaticDefense ?? DEFAULT_CONFIG.automaticDefense,
    origin: {
      latitude: finite(source.origin?.latitude, DEFAULT_CONFIG.origin.latitude, -90, 90),
      longitude: finite(source.origin?.longitude, DEFAULT_CONFIG.origin.longitude, -180, 180),
      altitudeM: finite(source.origin?.altitudeM, DEFAULT_CONFIG.origin.altitudeM),
    },
    defendedPoint: point(source.defendedPoint, DEFAULT_CONFIG.defendedPoint),
    sites: sites.map((site, siteIndex) => ({
      id: String(site.id ?? `site-${siteIndex + 1}`),
      position: point(site.position),
      ...(site.bankAssets?{bankAssets:structuredClone(site.bankAssets),placementIntent:structuredClone(site.placementIntent)}:{}),
      sensorRangeM: finite(site.sensorRangeM, 30000, 1),
      trackBuildSeconds: finite(site.trackBuildSeconds, 0.35, 0.01),
      readiness: ['constructing', 'maintenance'].includes(site.readiness) ? site.readiness : 'deployed',
      buildRemainingSeconds: finite(site.buildRemainingSeconds, 0, 0),
      layers: (Array.isArray(site.layers) ? site.layers : []).map((layer, layerIndex) => ({
        id: String(layer.id ?? `layer-${layerIndex + 1}`),
        kind: layer.kind === 'laser' ? 'laser' : 'interceptor',
        targetKinds: Array.isArray(layer.targetKinds) ? layer.targetKinds.map(String) : ['drone'],
        minRangeM: finite(layer.minRangeM, 0, 0),
        maxRangeM: finite(layer.maxRangeM, 5000, 1),
        minAltitudeM: finite(layer.minAltitudeM, 0, 0),
        maxAltitudeM: finite(layer.maxAltitudeM, 20000, 1),
        interceptorSpeedMps: finite(layer.interceptorSpeedMps, 500, 1),
        fuseRadiusM: finite(layer.fuseRadiusM, 50, 1),
        dwellSeconds: finite(layer.dwellSeconds, 1.4, 0.05),
        reloadSeconds: finite(layer.reloadSeconds, 1, 0),
        inventory: Math.floor(finite(layer.inventory, 1, 0)),
        effectiveness: finite(layer.effectiveness, 1, 0, 1),
        cooldownSeconds: finite(layer.cooldownSeconds, 0, 0),
      })),
    })),
  };
}

export class SimulationRoom {
  constructor(id = 'default', config = {}, persistedState = null) {
    this.id = String(id);
    if (persistedState == null) this.reset(config);
    else this.restore(persistedState);
  }

  reset(config = {}) {
    this.runId = randomUUID();
    this.entitySequence = 0;
    this.config = normalizedConfig(config);
    this.tick = 0;
    this.timeSeconds = 0;
    this.targets = new Map();
    this.infrastructure = new InfrastructureState();
    this.groundControls = new GroundControls();
    this.playerPresence = new PlayerPresence(this.id);
    this.engagements = new Map();
    this.interceptOrders = new Map();
    this.events = [];
    this.eventSequence = 0;
    this.statistics = { launched: 0, intercepted: 0, leaked: 0, missed: 0 };
    this.randomState = [...this.id].reduce((state, character) => Math.imul(state ^ character.charCodeAt(0), 16777619) >>> 0, 2166136261) || 1;
    this.record('simulation-reset', { origin: this.config.origin });
    this.vehicleSystem = new VehicleLogisticsSystem({
      onEvent: (type, data) => this.record(type, data),
    });
    for (const site of this.config.sites) this.vehicleSystem.addSitePackage(site);
    return this.snapshot();
  }

  restore(state) {
    if (state == null || typeof state !== 'object') throw new Error('persisted room state must be an object');
    if (state.storageVersion !== 1) throw new Error(`unsupported persisted room storage version: ${state.storageVersion}`);
    if (String(state.gameId) !== this.id) throw new Error(`persisted game id does not match room: ${this.id}`);
    this.runId = String(state.runId ?? '').trim();
    if (!this.runId) throw new Error(`persisted room is missing runId: ${this.id}`);
    this.config = normalizedConfig(state.config);
    this.tick = Math.floor(finite(state.tick, 0, 0));
    this.timeSeconds = finite(state.timeSeconds, 0, 0);
    this.targets = new Map((Array.isArray(state.targets) ? state.targets : []).map(target => [String(target.id), structuredClone(target)]));
    this.engagements = new Map((Array.isArray(state.engagements) ? state.engagements : []).map(engagement => [String(engagement.id), structuredClone(engagement)]));
    this.interceptOrders = new Map((Array.isArray(state.interceptOrders) ? state.interceptOrders : []).map(order => [String(order.targetId), structuredClone(order)]));
    this.events = Array.isArray(state.events) ? structuredClone(state.events) : [];
    this.eventSequence = Math.floor(finite(state.eventSequence, 0, 0));
    this.statistics = {
      launched: Math.floor(finite(state.statistics?.launched, 0, 0)),
      intercepted: Math.floor(finite(state.statistics?.intercepted, 0, 0)),
      leaked: Math.floor(finite(state.statistics?.leaked, 0, 0)),
      missed: Math.floor(finite(state.statistics?.missed, 0, 0)),
    };
    this.randomState = Math.floor(finite(state.randomState, 1, 1, 0xffffffff)) >>> 0;
    this.entitySequence = Math.floor(finite(state.entitySequence, 0, 0));
    this.vehicleSystem = new VehicleLogisticsSystem({
      onEvent: (type, data) => this.record(type, data),
    });
    this.vehicleSystem.restore(state.vehicles ?? []);
    this.infrastructure = new InfrastructureState(state.infrastructure ?? []);
    this.groundControls = new GroundControls(state.controlledVehicles ?? []);
    this.playerPresence = new PlayerPresence(this.id, state.players ?? []);
    for(const entity of this.infrastructure.entities.values()){
      if(entity.componentState&&protectedEntrance(this.id,entity.id))entity.componentState.paused=true;
    }
    return this.snapshot();
  }

  exportState({encodeSurface}={}) {
    return structuredClone({
      storageVersion: 1,
      controlledVehicles: this.groundControls.exportState({encodeSurface}),
      players: this.playerPresence.exportState(),
      infrastructure: this.infrastructure.snapshot(),
      gameId: this.id,
      runId: this.runId,
      config: this.config,
      tick: this.tick,
      timeSeconds: this.timeSeconds,
      targets: [...this.targets.values()],
      engagements: [...this.engagements.values()],
      interceptOrders: [...this.interceptOrders.values()],
      eventSequence: this.eventSequence,
      statistics: this.statistics,
      randomState: this.randomState,
      entitySequence: this.entitySequence,
      vehicles: this.vehicleSystem.snapshot(),
    });
  }

  nextEntityId(kind) {
    this.entitySequence += 1;
    const hex = createHash('sha256')
      .update(`${this.runId}:${String(kind)}:${this.entitySequence}`)
      .digest('hex')
      .slice(0, 32)
      .split('');
    hex[12] = '5';
    hex[16] = ['8', '9', 'a', 'b'][Number.parseInt(hex[16], 16) % 4];
    const value = hex.join('');
    return `${value.slice(0, 8)}-${value.slice(8, 12)}-${value.slice(12, 16)}-${value.slice(16, 20)}-${value.slice(20)}`;
  }

  nextRandom() {
    let value = this.randomState;
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    this.randomState = value >>> 0;
    return this.randomState / 0x100000000;
  }

  record(type, data = {}) {
    const event = { sequence: ++this.eventSequence, tick: this.tick, timeSeconds: this.timeSeconds, type, ...data };
    this.events.push(event);
    if (this.events.length > 2000) this.events.splice(0, this.events.length - 2000);
    return event;
  }

  spawnTarget(input = {}) {
    const id = String(input.id ?? this.nextEntityId('target'));
    if (this.targets.has(id)) throw new Error(`target already exists: ${id}`);
    const start = point(input.start, { x: -9000, y: 5200, z: 750 });
    const destination = point(input.destination, this.config.defendedPoint);
    const speedMps = finite(input.speedMps, 76, 1, 3000);
    const unit = direction(start, destination);
    const target = {
      id,
      assetType: 'target',
      catalogId: String(input.catalogId ?? 'shahed'),
      kind: String(input.kind ?? 'drone'),
      label: String(input.label ?? 'OWA drone'),
      position: start,
      start,
      destination,
      velocity: { x: unit.x * speedMps, y: unit.y * speedMps, z: unit.z * speedMps },
      speedMps,
      trajectory: input.trajectory === 'ballistic' || input.kind === 'ballistic' ? 'ballistic' : 'direct',
      durationSeconds: finite(input.durationSeconds, distance(start, destination) / speedMps, 0.1, 86400),
      apexM: finite(input.apexM, 8000, 0, 200000),
      trajectoryElapsedSeconds: 0,
      status: 'active',
      trackQuality: 0,
      trackSeconds: 0,
      spawnedAt: this.timeSeconds,
      terminalAt: null,
    };
    this.targets.set(id, target);
    this.record('target-spawned', { targetId: id, catalogId: target.catalogId, position: { ...start } });
    return structuredClone(target);
  }

  deploySite(input = {}) {
    const id = String(input.id ?? `defense-site-${this.config.sites.length + 1}`);
    const existing=this.config.sites.find(site=>site.id===id);
    if(existing){
      if(input.bankAssets&&JSON.stringify(existing.bankAssets)===JSON.stringify(input.bankAssets)&&JSON.stringify(existing.placementIntent)===JSON.stringify(input.placementIntent))return structuredClone(existing);
      throw new Error(`site already exists with different terms: ${id}`);
    }
    if(input.bankAssets){
      const expected=['radar','command','resupply','recovery',...(input.layerIds??[]).map(id=>'launcher-'+id)];
      const ids=Object.values(input.bankAssets).map(asset=>asset.id);
      if(expected.length!==ids.length||expected.some(role=>!input.bankAssets[role])||new Set(ids).size!==ids.length||ids.some(id=>typeof id!=='string'||!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id)||this.vehicleSystem.vehicles.has(id)))throw Error('Distinct bank UUIDs required for every site component');
    }
    const requestedLayerIds = Array.isArray(input.layerIds) && input.layerIds.length > 0
      ? input.layerIds.map(String)
      : ['point-defense'];
    const templates = DEFAULT_CONFIG.sites[0].layers.filter(layer => requestedLayerIds.includes(layer.id));
    if (templates.length !== requestedLayerIds.length) throw new Error('one or more requested defense layers are unknown');
    const buildSeconds = finite(input.buildSeconds, 15, 0, 86400);
    const site = normalizedConfig({ sites: [{
      id,
      position: point(input.position),
      bankAssets:input.bankAssets, placementIntent:input.placementIntent,
      sensorRangeM: finite(input.sensorRangeM, 15000, 1),
      readiness: buildSeconds > 0 ? 'constructing' : 'deployed',
      buildRemainingSeconds: buildSeconds,
      layers: templates,
    }] }).sites[0];
    site.readiness = buildSeconds > 0 ? 'constructing' : 'deployed';
    site.buildRemainingSeconds = buildSeconds;
    this.config.sites.push(site);
    this.vehicleSystem.addSitePackage(site);
    this.record('defense-site-construction-started', { siteId: id, position: { ...site.position }, buildSeconds, layerIds: requestedLayerIds });
    if (buildSeconds === 0) this.record('defense-site-deployed', { siteId: id, layerIds: requestedLayerIds });
    return structuredClone(site);
  }

  reportVehicle(input = {}) {
    if(this.groundControls.vehicles.has(input.id)||this.groundControls.snapshot().some(v=>v.terrainAssetId===input.id))throw Error('server-controlled vehicle rejects viewer position writes');
    return this.vehicleSystem.reportPosition(input);
  }

  defenseSensors() {
    return this.config.sites.flatMap(site=>[...this.vehicleSystem.vehicles.values()]
      .filter(v=>v.siteId===site.id&&v.role==='sensor').map(v=>({id:v.id,siteId:site.id,
        status:v.status,operational:site.readiness==='deployed'&&v.status==='deployed',
        rangeM:site.sensorRangeM,trackBuildSeconds:site.trackBuildSeconds})));
  }

  spawnIncoming({incomingType,requestId,destination,destinationCoordinates,headingDeg,approachDistanceM,altitudeM,speedMps}) {
    if(!['drone','cruise','ballistic'].includes(incomingType))throw Error('Unknown synthetic incoming type');
    if(typeof requestId!=='string'||!requestId.trim()||requestId.length>100)throw Error('requestId must contain 1..100 characters');
    if(!destination||!['x','y','z'].every(k=>Number.isFinite(destination[k])))throw Error('Explicit finite destination required');
    if(!Number.isFinite(headingDeg)||headingDeg<0||headingDeg>=360||!Number.isFinite(approachDistanceM)||approachDistanceM<10||approachDistanceM>100000||!Number.isFinite(altitudeM)||altitudeM<10||altitudeM>20000||!Number.isFinite(speedMps)||speedMps<1||speedMps>3000)throw Error('Invalid synthetic incoming parameters');
    const terms={incomingType,destinationCoordinates,headingDeg,approachDistanceM,altitudeM,speedMps};
    const id='scenario-incoming:'+requestId,existing=this.targets.get(id);
    if(existing){if(JSON.stringify(existing.scenarioTerms)!==JSON.stringify(terms))throw Error('requestId already identifies different scenario terms');return {target:structuredClone(existing),alreadyExists:true};}
    const angle=headingDeg*Math.PI/180,start={x:destination.x-Math.sin(angle)*approachDistanceM,y:destination.y-Math.cos(angle)*approachDistanceM,z:destination.z+altitudeM};
    this.spawnTarget({id,kind:incomingType,catalogId:incomingType==='ballistic'?'ballistic':'shahed',label:'Synthetic '+incomingType,
      start,destination,speedMps,durationSeconds:distance(start,destination)/speedMps,apexM:0});
    const target=this.targets.get(id);target.scenarioTerms=terms;
    return {target:structuredClone(target),alreadyExists:false,testOnly:true};
  }

  spawnTestIncoming({incomingType,requestId,siteId,testLayerId,headingDeg=90}) {
    if(!Number.isFinite(headingDeg)||headingDeg<0||headingDeg>=360)throw Error('headingDeg must be in [0, 360): north=0, east=90, south=180, west=270');
    if(!['drone','cruise','ballistic'].includes(incomingType))throw Error('incomingType must be drone, cruise or ballistic');
    if(typeof requestId!=='string'||!requestId.trim()||requestId.length>100)throw Error('requestId must contain 1..100 characters');
    const site=this.config.sites.find(s=>s.id===siteId),layer=site?.layers.find(l=>l.id===testLayerId);
    if(!layer)throw Error('Choose siteId/testLayerId from the current test cases');
    if(!layer.targetKinds.includes(incomingType))throw Error('Incoming type is not supported by this fictional test layer');
    const id='test-incoming:'+requestId,terms={incomingType,siteId,testLayerId,headingDeg},existing=this.targets.get(id);
    if(existing){if(JSON.stringify(existing.testCase)!==JSON.stringify(terms))throw Error('requestId already identifies different test terms');return {target:structuredClone(existing),alreadyExists:true};}
    // Synthetic fixture placement only. Does not select or activate an engagement.
    const maxRange=Math.min(layer.maxRangeM,site.sensorRangeM),altitude=(layer.minAltitudeM+layer.maxAltitudeM)/2;
    const vertical=altitude-site.position.z,minRange=Math.max(layer.minRangeM,Math.abs(vertical));
    if(maxRange<=minRange)throw Error('No test fixture fits this configured game layer and sensor');
    const range=(minRange+maxRange)/2,horizontal=Math.sqrt(range*range-vertical*vertical);
    const angle=headingDeg*Math.PI/180;
    const start={x:site.position.x-Math.sin(angle)*horizontal,y:site.position.y-Math.cos(angle)*horizontal,z:altitude};
    const destination={...site.position,z:altitude};
    this.spawnTarget({id,kind:incomingType,catalogId:incomingType==='ballistic'?'ballistic':'shahed',
      label:'Synthetic '+incomingType+' test',start,destination,speedMps:horizontal/120,durationSeconds:120,apexM:0});
    const target=this.targets.get(id);target.testCase=terms;
    return {target:structuredClone(target),alreadyExists:false,testOnly:true,
      note:'Synthetic 120-second fixture using configured game bounds. Not a realistic trajectory or an activation command.'};
  }

  setDefenseMode(mode) {
    if (!['functions', 'automatic'].includes(mode)) throw Error('mode must be functions or automatic');
    const cancelledOrders = mode === 'functions' ? this.interceptOrders.size : 0;
    if (mode === 'functions') this.interceptOrders.clear();
    this.config.automaticDefense = mode === 'automatic';
    this.record('defense-mode-changed', {mode, cancelledOrders});
    return {mode, cancelledOrders, activeEngagements: [...this.engagements.values()].filter(e=>e.status==='active').length};
  }

  defenseObservation() {
    const active = [...this.engagements.values()].filter(e=>e.status==='active');
    return structuredClone({gameId:this.id, tick:this.tick,
      mode:this.config.automaticDefense?'automatic':'functions', lastEventSequence:this.eventSequence, sensors:this.defenseSensors(),
      tracks:[...this.targets.values()].filter(t=>t.trackQuality>0).map(t=>{
        const engagement=active.find(e=>e.targetId===t.id);
        const ready=t.status==='active'&&t.trackQuality>=1&&!engagement;
        return {id:t.id, label:t.label, kind:t.kind, headingDeg:t.scenarioTerms?.headingDeg??t.testCase?.headingDeg??null, destinationCoordinates:t.scenarioTerms?.destinationCoordinates??null, detectedBy:t.detectedBy??[], state:t.status!=='active'?t.status:engagement?'engaged':t.trackQuality>=1?'tracked':'detecting',
          availableActions:ready?this.eligibleLayers(t).map(({site,layer})=>({siteId:site.id,layerId:layer.id,kind:layer.kind})):[]};
      }),
      layers:this.config.sites.flatMap(site=>site.layers.map(layer=>({siteId:site.id,layerId:layer.id,kind:layer.kind,
        targetKinds:layer.targetKinds,readiness:site.readiness,remainingShots:layer.kind==='laser'?null:layer.inventory,
        available:layer.inventory>0&&layer.cooldownSeconds===0&&site.readiness==='deployed',cooldownSeconds:layer.cooldownSeconds}))),
      engagements:active.map(e=>({id:e.id,targetId:e.targetId,siteId:e.siteId,layerId:e.layerId,status:e.status})),statistics:this.statistics});
  }

  eligibleLayers(target, requestedLayerId = null, requestedSiteId = null) {
    const candidates = [];
    for (const site of this.config.sites) {
      if (site.readiness !== 'deployed' || (requestedSiteId !== null && site.id !== requestedSiteId)) continue;
      const rangeM = distance(site.position, target.position);
      for (const layer of site.layers) {
        if (requestedLayerId != null && layer.id !== requestedLayerId) continue;
        if (!layer.targetKinds.includes(target.kind) || layer.inventory <= 0 || layer.cooldownSeconds > 0) continue;
        if (rangeM < layer.minRangeM || rangeM > layer.maxRangeM) continue;
        if (target.position.z < layer.minAltitudeM || target.position.z > layer.maxAltitudeM) continue;
        candidates.push({ site, layer, rangeM });
      }
    }
    candidates.sort((a, b) => b.rangeM - a.rangeM);
    return candidates;
  }

  commandIntercept({ targetId, layerId = null, siteId = null } = {}, { queueIfUntracked = true } = {}) {
    const target = targetId == null
      ? [...this.targets.values()].find(candidate => candidate.status === 'active')
      : this.targets.get(String(targetId));
    if (target == null) return { accepted: false, reason: 'target-not-found' };
    if (target.status !== 'active') return { accepted: false, reason: 'target-not-active', targetId: target.id };
    if ([...this.engagements.values()].some(item => item.targetId === target.id && item.status === 'active')) {
      return { accepted: false, reason: 'already-engaged', targetId: target.id };
    }
    if (target.trackQuality < 1) {
      if (!queueIfUntracked) return { accepted: false, reason: 'target-not-tracked', targetId: target.id };
      this.interceptOrders.set(target.id, { targetId: target.id, layerId, siteId });
      this.record('intercept-authorized', { targetId: target.id, layerId, queued: true });
      return { accepted: true, queued: true, targetId: target.id };
    }
    const selected = this.eligibleLayers(target, layerId, siteId)[0];
    if (selected == null) return { accepted: false, reason: 'no-ready-layer', targetId: target.id };
    const { site, layer } = selected;
    if (layer.kind !== 'laser') layer.inventory -= 1;
    layer.cooldownSeconds = layer.reloadSeconds;
    const id = this.nextEntityId('engagement');
    const engagement = {
      id,
      assetType: 'interceptor',
      targetId: target.id,
      siteId: site.id,
      layerId: layer.id,
      kind: layer.kind,
      position: { ...site.position },
      velocity: { x: 0, y: 0, z: 0 },
      speedMps: layer.interceptorSpeedMps,
      fuseRadiusM: layer.fuseRadiusM,
      maxRangeM: layer.maxRangeM,
      dwellSeconds: layer.dwellSeconds,
      dwellElapsedSeconds: 0,
      effectiveness: layer.effectiveness,
      status: 'active',
      launchedAt: this.timeSeconds,
    };
    this.engagements.set(id, engagement);
    this.interceptOrders.delete(target.id);
    this.statistics.launched += 1;
    this.record('interceptor-launched', { engagementId: id, targetId: target.id, siteId: site.id, layerId: layer.id });
    return { accepted: true, engagement: structuredClone(engagement) };
  }

  updateTracksAndAutomation(dt) {
    for (const target of this.targets.values()) {
      if (target.status !== 'active') continue;
      const previousQuality=target.trackQuality;
      const working=this.defenseSensors().filter(sensor=>sensor.operational);
      const sensors = this.config.sites.filter(site => working.some(sensor=>sensor.siteId===site.id) && distance(site.position, target.position) <= site.sensorRangeM);
      target.detectedBy=working.filter(sensor=>sensors.some(site=>site.id===sensor.siteId)).map(sensor=>({sensorId:sensor.id,siteId:sensor.siteId}));
      if (sensors.length > 0) {
        target.trackSeconds += dt;
        const fastestTrack = Math.min(...sensors.map(site => site.trackBuildSeconds));
        target.trackQuality = Math.min(1, target.trackSeconds / fastestTrack);
      } else {
        target.trackSeconds = Math.max(0, target.trackSeconds - dt);
        target.trackQuality = 0;
      }
      if(previousQuality===0&&target.trackQuality>0)this.record('target-detected',{targetId:target.id,kind:target.kind,detectedBy:target.detectedBy});
      if(previousQuality<1&&target.trackQuality>=1)this.record('target-tracked',{targetId:target.id,kind:target.kind,detectedBy:target.detectedBy});
      if(previousQuality>0&&target.trackQuality===0)this.record('target-track-lost',{targetId:target.id});
      if (target.trackQuality < 1) continue;
      const order = this.interceptOrders.get(target.id);
      if (!this.config.automaticDefense && order == null) continue;
      this.commandIntercept(order ?? { targetId: target.id }, { queueIfUntracked: false });
    }
  }

  step(dt = 1 / this.config.tickRateHz) {
    const seconds = finite(dt, 1 / this.config.tickRateHz, 0.0001, 1);
    this.tick += 1;
    this.timeSeconds += seconds;
    this.infrastructure.step(seconds);
    this.playerPresence.reconcileActions(this.infrastructure);
    this.groundControls.step(seconds,(v,from,to)=>movementDenial(this.infrastructure.entities.values(),v.ownerAccountId,vehicleWorldPosition(v,from,this.config.origin),vehicleWorldPosition(v,to,this.config.origin)));
    this.playerPresence.step(seconds,(p,from,to)=>movementDenial(this.infrastructure.entities.values(),p.ownerAccountId,from,to));
    for (const site of this.config.sites) {
      if (site.readiness === 'constructing') {
        site.buildRemainingSeconds = Math.max(0, site.buildRemainingSeconds - seconds);
        if (site.buildRemainingSeconds === 0) {
          site.readiness = 'deployed';
          this.vehicleSystem.setSiteReadiness(site.id, 'deployed');
          this.record('defense-site-deployed', { siteId: site.id, layerIds: site.layers.map(layer => layer.id) });
        }
      }
      for (const layer of site.layers) layer.cooldownSeconds = Math.max(0, layer.cooldownSeconds - seconds);
    }
    for (const target of this.targets.values()) {
      if (target.status !== 'active') continue;
      let moved;
      if (target.trajectory === 'ballistic') {
        target.trajectoryElapsedSeconds += seconds;
        const progress = Math.min(1, target.trajectoryElapsedSeconds / target.durationSeconds);
        const previousPosition = target.position;
        const position = {
          x: target.start.x + (target.destination.x - target.start.x) * progress,
          y: target.start.y + (target.destination.y - target.start.y) * progress,
          z: target.start.z + (target.destination.z - target.start.z) * progress + 4 * target.apexM * progress * (1 - progress),
        };
        target.velocity = {
          x: (position.x - previousPosition.x) / seconds,
          y: (position.y - previousPosition.y) / seconds,
          z: (position.z - previousPosition.z) / seconds,
        };
        moved = { position, arrived: progress >= 1 };
      } else {
        moved = moveToward(target.position, target.destination, target.speedMps * seconds);
      }
      target.position = moved.position;
      if (target.trajectory !== 'ballistic') {
        const unit = direction(target.position, target.destination);
        target.velocity = moved.arrived ? { x: 0, y: 0, z: 0 } : { x: unit.x * target.speedMps, y: unit.y * target.speedMps, z: unit.z * target.speedMps };
      }
      if (moved.arrived) {
        target.status = 'leaked';
        target.terminalAt = this.timeSeconds;
        this.statistics.leaked += 1;
        this.record('target-leaked', { targetId: target.id, position: { ...target.position } });
      }
    }
    this.updateTracksAndAutomation(seconds);
    for (const engagement of this.engagements.values()) {
      if (engagement.status !== 'active') continue;
      const target = this.targets.get(engagement.targetId);
      if (target == null || target.status !== 'active') {
        engagement.status = 'expended';
        continue;
      }
      if (engagement.kind === 'laser') {
        if (distance(engagement.position, target.position) > engagement.maxRangeM) {
          engagement.status = 'missed';
          this.statistics.missed += 1;
          this.record('interceptor-missed', { targetId: target.id, engagementId: engagement.id, reason: 'laser-range' });
          continue;
        }
        engagement.dwellElapsedSeconds += seconds;
        if (engagement.dwellElapsedSeconds < engagement.dwellSeconds) continue;
        const success = this.nextRandom() <= engagement.effectiveness;
        engagement.status = success ? 'expended' : 'missed';
        if (success) {
          target.status = 'intercepted';
          target.terminalAt = this.timeSeconds;
          this.statistics.intercepted += 1;
          this.record('target-intercepted', { targetId: target.id, engagementId: engagement.id, layerId: engagement.layerId, position: { ...target.position } });
        } else {
          this.statistics.missed += 1;
          this.record('interceptor-missed', { targetId: target.id, engagementId: engagement.id, reason: 'laser-dwell' });
        }
        continue;
      }
      const unit = direction(engagement.position, target.position);
      engagement.velocity = { x: unit.x * engagement.speedMps, y: unit.y * engagement.speedMps, z: unit.z * engagement.speedMps };
      const moved = moveToward(engagement.position, target.position, engagement.speedMps * seconds);
      engagement.position = moved.position;
      if (moved.arrived || distance(engagement.position, target.position) <= engagement.fuseRadiusM) {
        // Effectiveness remains visible in state. The proof runtime resolves the
        // engagement deterministically so replays do not depend on wall-clock RNG.
        const success = this.nextRandom() <= engagement.effectiveness;
        engagement.status = success ? 'expended' : 'missed';
        if (success) {
          target.status = 'intercepted';
          target.terminalAt = this.timeSeconds;
          this.statistics.intercepted += 1;
          this.record('target-intercepted', { targetId: target.id, engagementId: engagement.id, layerId: engagement.layerId, position: { ...target.position } });
        } else {
          this.statistics.missed += 1;
          this.record('interceptor-missed', { targetId: target.id, engagementId: engagement.id });
        }
      }
    }
    this.vehicleSystem.syncEngagements(new Set(
      [...this.engagements.values()].filter(item => item.status === 'active').map(item => item.siteId),
    ));
    this.vehicleSystem.step(seconds);
    return this.snapshot();
  }

  counts() {
    const targets = [...this.targets.values()];
    const engagements = [...this.engagements.values()];
    const sites = this.config.sites;
    const vehicleCounts = this.vehicleSystem.counts();
    return {
      all: targets.length + engagements.length + sites.length + vehicleCounts.total,
      vehicles: vehicleCounts,
      targets: { total: targets.length, active: targets.filter(item => item.status === 'active').length, intercepted: this.statistics.intercepted, leaked: this.statistics.leaked },
      defense: {
        sites: sites.length,
        deployedSites: sites.filter(item => item.readiness === 'deployed').length,
        constructingSites: sites.filter(item => item.readiness === 'constructing').length,
        maintenanceSites: sites.filter(item => item.readiness === 'maintenance').length,
        activeInterceptors: engagements.filter(item => item.status === 'active').length,
      },
    };
  }

  snapshot() {
    return structuredClone({
      protocol: 'atlantis-simulation-v1',
      automaticDefense: this.config.automaticDefense,
      defenseSensors: this.defenseSensors(),
      controlledVehicles: this.groundControls.snapshot(),
      players: this.playerPresence.snapshot(),
      infrastructure: [...this.infrastructure.snapshot(),...this.vehicleSystem.snapshot().filter(v=>v.bankAssetId).map(v=>({
        id:v.bankAssetId,modelId:v.modelId,label:InfrastructureState.modelLabel(v.modelId),equipmentState:v.equipmentState,position:v.position,headingDeg:v.headingDeg,sourceVehicleId:v.id,
        presentation:v.modelId.startsWith('support-')?'infrastructure':'defense-site',simulationScope:'defense-site-component',siteId:v.siteId,ownerAccountId:v.ownerAccountId,
      }))].map(record=>{
        const {accessPolicy,...entity}=record;
        if(accessPolicy)entity.accessControl={protected:true,interactionRadiusM:accessPolicy.interactionRadiusM};
        const mobile=this.groundControls.vehicles.get(entity.id);
        if(mobile?.presentation==='infrastructure'){
          const state=this.groundControls.observe(entity.id),origin=this.config.origin;
          const position={x:(state.lon-origin.longitude)*Math.PI/180*6378137*Math.cos(origin.latitude*Math.PI/180),y:(state.lat-origin.latitude)*Math.PI/180*6378137,z:state.position.z+mobile.renderOffsetM-origin.altitudeM};
          const rig=structuredClone(entity.equipmentState);
          if(rig?.mechanisms.tracks){rig.mechanisms.tracks.phase=state.signedDistanceM;rig.mechanisms.tracks.actual.track_speed_mps=state.speedMps;rig.mechanisms.tracks.actual.steering_deg=state.steeringRad*180/Math.PI;}
          if(rig?.mechanisms.propulsion){rig.mechanisms.propulsion.phase=state.signedDistanceM;rig.mechanisms.propulsion.actual.shaft_rpm=state.speedMps*6;}
          return {...entity,position,headingDeg:-state.headingRad*180/Math.PI,equipmentState:rig,movement:state,simulationScope:'mobile-asset'};
        }
        if(entity.siteId){
          const site=this.config.sites.find(s=>s.id===entity.siteId);
          if(!site)throw Error('Defense component has no site');
          const role=Object.entries(site.bankAssets).find(([,asset])=>asset.id===entity.id)?.[0];
          const supports={'command':[-25,-5,0],'resupply':[0,-5,0],'recovery':[25,-5,0]};
          let offset;
          if(role==='radar')offset=[0,-35,0];
          else if(Object.hasOwn(supports,role))offset=supports[role];
          else {const index=site.layers.findIndex(layer=>'launcher-'+layer.id===role);if(index<0)throw Error('Unknown defense component role');offset=[index*35-(site.layers.length-1)*17.5,35,3];}
          // Existing display pad layout, not navigation or a tactical siting model.
          entity.position={x:site.position.x+offset[0],y:site.position.y+offset[1],z:site.position.z+offset[2]};
        }
        if(!entity.sourceVehicleId)return entity;
        const vehicle=this.vehicleSystem.vehicles.get(entity.sourceVehicleId);
        if(!vehicle)return {...entity,bindingError:'source-vehicle-missing'};
        return {...entity,position:entity.siteId?entity.position:{...vehicle.position},headingDeg:vehicle.headingDeg,
          operationalState:{status:vehicle.status,condition:vehicle.condition,fuel:structuredClone(vehicle.fuel),recoveryRemainingSeconds:vehicle.recoveryRemainingSeconds}};
      }),
      gameId: this.id,
      runId: this.runId,
      tick: this.tick,
      tickRateHz: this.config.tickRateHz,
      timeSeconds: this.timeSeconds,
      origin: this.config.origin,
      targets: [...this.targets.values()],
      engagements: [...this.engagements.values()].filter(item => item.status === 'active'),
      sites: this.config.sites,
      vehicles: this.vehicleSystem.snapshot(),
      statistics: this.statistics,
      counts: this.counts(),
      lastEventSequence: this.eventSequence,
    });
  }

  eventsAfter(sequence = 0) {
    const after = finite(sequence, 0, 0);
    return structuredClone(this.events.filter(event => event.sequence > after));
  }
}

export class SimulationEngine {
  constructor({ tickRateHz = 30 } = {}) {
    this.rooms = new Map();
    this.tickRateHz = finite(tickRateHz, 30, 1, 120);
  }
  room(id = 'default') {
    const key = String(id);
    if (!this.rooms.has(key)) this.rooms.set(key, new SimulationRoom(key, { tickRateHz: this.tickRateHz }));
    return this.rooms.get(key);
  }
  reset(id, config) {
    const room = new SimulationRoom(id, { ...config, tickRateHz: this.tickRateHz });
    this.rooms.set(String(id), room);
    return room;
  }
  restore(id, state) {
    const key = String(id);
    const room = new SimulationRoom(key, {}, state);
    this.rooms.set(key, room);
    return room;
  }
  step(dt) { for (const room of this.rooms.values()) room.step(dt); }
}
