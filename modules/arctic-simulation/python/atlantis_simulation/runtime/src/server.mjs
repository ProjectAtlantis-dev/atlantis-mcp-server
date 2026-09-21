import {equipmentContract} from './equipment-state.mjs';
import http from 'node:http';
import { parseArgs } from 'node:util';
import { SimulationEngine } from './engine.mjs';
import { SimulationStore } from './simulation-store.mjs';
import { InfrastructureState } from './infrastructure-state.mjs';
import { FixedClock } from './fixed-clock.mjs';
import {physicalSubject,interactionDecision} from './infrastructure-access.mjs';
import {protectedEntrance,playerPolicy} from './player-presence.mjs';

const { values } = parseArgs({ options: {
  host: { type: 'string', default: process.env.ATLANTIS_SIM_HOST ?? '127.0.0.1' },
  port: { type: 'string', default: process.env.ATLANTIS_SIM_PORT ?? '5190' },
  token: { type: 'string', default: process.env.ATLANTIS_SIM_TOKEN ?? '' },
  'tick-rate': { type: 'string', default: process.env.ATLANTIS_SIM_TICK_RATE ?? '30' },
  database: { type: 'string', default: process.env.ATLANTIS_SIM_DB_PATH ?? '' },
} });
const host = values.host;
const port = Number(values.port);
const token = values.token;
const tickRateHz = Math.max(1, Math.min(120, Number(values['tick-rate']) || 30));
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('invalid --port');
if (!token) throw new Error('--token is required');
if (!values.database) throw new Error('--database is required');

const engine = new SimulationEngine({ tickRateHz });
const store = new SimulationStore(values.database);
const restoredRooms = store.restoreEngine(engine);
let previous = process.hrtime.bigint();
const clock = new FixedClock({hz:tickRateHz,maxStepsPerPump:8});
const timer = setInterval(() => {
  const now = process.hrtime.bigint();
  const elapsedSeconds = Number(now - previous) / 1e9;
  previous = now;
  clock.advance(elapsedSeconds,dt=>{
    engine.step(dt);
    // Commit before another command/observation can expose this completed step.
    // SQLite uses WAL + synchronous=FULL. Failure stops the authority instead
    // of serving motion that was never durably recorded.
    store.saveEngine(engine);
  });
}, Math.max(2, Math.floor(1000 / tickRateHz / 2)));
timer.unref();
const checkpointTimer = setInterval(() => store.saveEngine(engine), 1000);
checkpointTimer.unref();

function send(response, status, body) {
  const payload = Buffer.from(JSON.stringify(body));
  response.writeHead(status, { 'content-type': 'application/json', 'content-length': payload.length, 'cache-control': 'no-store' });
  response.end(payload);
}

async function readJson(request) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > 4 * 1024 * 1024) throw new Error('request body too large');
    chunks.push(chunk);
  }
  if (size === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host ?? 'localhost'}`);
    if (url.pathname === '/health' && request.method === 'GET') {
      send(response, 200, {
        ok: true,
        protocol: 'atlantis-simulation-v1',
        tickRateHz,
        clock: {owner:'mcp-supervised-simulation',pendingSeconds:clock.pendingSeconds,dropsElapsedTime:false},
        rooms: engine.rooms.size,
        persistence: { driver: 'sqlite', restoredRooms },
      });
      return;
    }
    if (request.headers.authorization !== `Bearer ${token}`) {
      send(response, 401, { error: 'unauthorized' });
      return;
    }
    const match = url.pathname.match(/^\/games\/([^/]+)\/(snapshot|events|reset|targets|sites|vehicle-reports|intercept|infrastructure|infrastructure-catalog|component-contract|component-command|infrastructure-access|vehicle-control|player-control|defense-mode|defense-observation|test-incoming|scenario-incoming|equipment-contract|equipment-command)$/);
    if (match == null) {
      send(response, 404, { error: 'not-found' });
      return;
    }
    const gameId = decodeURIComponent(match[1]);
    const action = match[2];
    if(action==='scenario-incoming'&&request.method==='POST'){
      const room=engine.room(gameId),result=room.spawnIncoming(await readJson(request));
      store.saveRoom(room);send(response,result.alreadyExists?200:201,result);return;
    }
    if(action==='test-incoming'&&request.method==='POST'){
      const room=engine.room(gameId),result=room.spawnTestIncoming(await readJson(request));
      store.saveRoom(room);send(response,result.alreadyExists?200:201,result);return;
    }
    if(action==='defense-mode'&&request.method==='POST'){
      const room=engine.room(gameId),payload=await readJson(request);
      const result=room.setDefenseMode(payload.mode);store.saveRoom(room);send(response,200,result);return;
    }
    if(action==='defense-observation'&&request.method==='GET'){
      send(response,200,engine.room(gameId).defenseObservation());return;
    }
    if(action==='player-control'){
      if(request.method!=='POST'){send(response,405,{error:'method-not-allowed'});return;}
      const room=engine.room(gameId),payload=await readJson(request),operation=payload.operation;
      if(!['attach','claim','move','release','observe','requestEntry','action'].includes(operation))throw Error('Unknown player operation');
      const players=room.playerPresence;
      players.reconcileActions(room.infrastructure);
      const result=operation==='observe'?players.observe(payload.id):operation==='action'?players.action(payload.id):
        operation==='requestEntry'?players.requestEntry(payload,room.infrastructure):players[operation](payload);
      if(!['observe','action'].includes(operation)){room.record(`player-${operation}`,{id:payload.id});store.saveRoom(room);}
      send(response,200,result);return;
    }
    if(action==='vehicle-control'){
      if(request.method!=='POST'){send(response,405,{error:'method-not-allowed'});return;}
      const room=engine.room(gameId),payload=await readJson(request),operation=payload.operation;
      if(!['attach','claim','drive','release','observe','drive_to','sail_to','fly_to','mission_control','mission_status','capabilities','mission_surface'].includes(operation))throw Error('unknown vehicle control operation');
      if(operation==='attach'&&room.vehicleSystem.vehicles.has(payload.id))throw Error('vehicle already has a logistics state writer');
      if(operation==='attach'&&payload.presentation==='infrastructure'){
        if(!room.infrastructure.entities.has(payload.id))throw Error('Placed infrastructure instance required');
        if(!Number.isFinite(payload.renderOffsetM))throw Error('Finite authored model offset required');
      }
      let result;
      try {
        result=operation==='observe'?room.groundControls.observe(payload.id):room.groundControls[operation](payload);
      } catch(error) {
        send(response,409,{error:'vehicle_command_rejected',message:error.message});return;
      }
      if(operation==='attach'&&payload.presentation==='infrastructure'){
        const vehicle=room.groundControls.get(payload.id);
        vehicle.presentation='infrastructure';vehicle.renderOffsetM=payload.renderOffsetM;
        result=room.groundControls.observe(payload.id);
      }
      if(!['observe','mission_status','capabilities','mission_surface'].includes(operation))room.record(`vehicle-${operation}`,{id:payload.id});
      store.saveRoom(room);
      send(response,200,result);return;
    }
    if(action==='equipment-contract'&&request.method==='POST'){
      const payload=await readJson(request);send(response,200,equipmentContract(payload.modelId));return;
    }
    if(action==='equipment-command'&&request.method==='POST'){
      const room=engine.room(gameId),payload=await readJson(request);
      const placed=room.infrastructure.entities.get(payload.id),siteVehicle=room.vehicleSystem.vehicles.get(payload.id);
      const entity=placed??(siteVehicle?.bankAssetId?siteVehicle:null);
      if(!entity)throw Error('Unknown equipment instance');
      const mechanism=equipmentContract(entity.modelId).mechanisms.find(m=>m.id===payload.mechanismId);
      if(!mechanism)throw Error('Unknown mechanism');
      if(room.groundControls.vehicles.has(payload.id)&&['tracks','propulsion'].includes(mechanism.kind))throw Error('Movement controller owns this mechanism; use vehicle commands');
      const subject=mechanism.physicalAccess?physicalSubject(room,payload.subjectKind,payload.subjectId):null;
      if(!placed&&mechanism.physicalAccess)throw Error('Site equipment has no commissioned physical-access policy');
      const result=placed?room.infrastructure.equipment(payload,{accountId:payload.accountId,subject}):room.vehicleSystem.commandEquipment(payload);
      room.record('equipment-command',{id:payload.id,mechanismId:payload.mechanismId,revision:result.equipmentState.mechanisms[payload.mechanismId].revision});
      store.saveRoom(room);send(response,200,{accepted:true,entity:result});return;
    }
    if(action==='component-contract'&&request.method==='GET'){send(response,200,InfrastructureState.componentContract());return;}
    if(action==='infrastructure-access'){
      if(request.method!=='POST'){send(response,405,{error:'method-not-allowed'});return;}
      const room=engine.room(gameId),payload=await readJson(request),entity=room.infrastructure.entities.get(payload.id);
      if(!entity)throw Error('Unknown infrastructure id');
      if(payload.operation==='configure'){
        if(payload.accountId!==payload.policy?.ownerAccountId)throw Error('Policy owner must match authenticated account');
        const result=room.infrastructure.configureAccess(payload);store.saveRoom(room);send(response,200,{entity:result});return;
      }
      if(payload.operation==='discover'){
        const subjects=(payload.subjects??[]).map(s=>{const subject=physicalSubject(room,s.kind,s.id);return {...s,...interactionDecision(entity,subject,payload.accountId)};});
        send(response,200,{protected:!!entity.accessPolicy,subjects});return;
      }
      if(payload.operation!=='command')throw Error('Unknown infrastructure access operation');
      if(!entity.accessPolicy)throw Error('Access policy has not been commissioned');
      const subject=physicalSubject(room,payload.subjectKind,payload.subjectId);
      const result=room.infrastructure.command(payload,{subject,accountId:payload.accountId});
      if(result.componentState?.paused){entity.componentState.paused=false;result.componentState.paused=false;}
      room.record('protected-component-command',{id:entity.id,action:payload.action,subjectId:subject.id});store.saveRoom(room);send(response,200,{accepted:true,entity:room.snapshot().infrastructure.find(e=>e.id===entity.id)});return;
    }
    if(action==='component-command'){
      if(request.method!=='POST'){send(response,405,{error:'method-not-allowed'});return;}
      const room=engine.room(gameId),payload=await readJson(request);
      if(protectedEntrance(gameId,payload.id))throw Error('Protected entrance requires authenticated player request');
      const entity=room.infrastructure.command(payload);room.record('component-command',{entity,action:payload.action});store.saveRoom(room);
      send(response,200,{accepted:true,entity});return;
    }
    if(action==='infrastructure-catalog'&&request.method==='GET'){send(response,200,InfrastructureState.catalog());return;}
    if(action==='infrastructure'){
      const room=engine.room(gameId);
      if(request.method==='GET'){send(response,200,{entities:room.snapshot().infrastructure});return;}
      const operations={POST:'place',PATCH:'move',DELETE:'remove'},operation=operations[request.method];
      if(!operation){send(response,405,{error:'method-not-allowed'});return;}
      const payload=await readJson(request);
      if(protectedEntrance(gameId,payload.id))throw Error('Protected entrance placement is operator-managed');
      if(room.groundControls.vehicles.has(payload.id))throw Error('Asset has an attached movement controller; use vehicle commands');
      if(operation==='place'&&payload.sourceVehicleId&&!room.vehicleSystem.vehicles.has(payload.sourceVehicleId))throw Error('unknown sourceVehicleId');
      if(operation==='place'&&payload.sourceVehicleId&&room.vehicleSystem.vehicles.get(payload.sourceVehicleId).geodetic)throw Error('binding currently requires a local-ENU simulation vehicle');
      const entity=room.infrastructure[operation](payload);
      room.record(`infrastructure-${operation}`,{entity});store.saveRoom(room);
      send(response,request.method==='POST'?201:200,{entity});return;
    }
    if (request.method === 'GET' && action === 'snapshot') {
      const room = engine.room(gameId);
      store.saveRoom(room);
      send(response, 200, room.snapshot());
    } else if (request.method === 'GET' && action === 'events') {
      send(response, 200, { events: engine.room(gameId).eventsAfter(url.searchParams.get('after')) });
    } else if (request.method === 'POST' && action === 'reset') {
      if(engine.room(gameId).groundControls.vehicles.size)throw Error('Reset forbidden for commissioned Terrain vehicles');
      if(engine.room(gameId).playerPresence.players.size||(process.env.ATLANTIS_PLAYER_DEPLOYMENTS&&Object.keys(playerPolicy(gameId,false).players??{}).length))throw Error('Reset forbidden for commissioned player worlds');
      const room = engine.reset(gameId, await readJson(request));
      store.saveRoom(room);
      send(response, 200, room.snapshot());
    } else if (request.method === 'POST' && action === 'targets') {
      const room = engine.room(gameId);
      const target = room.spawnTarget(await readJson(request));
      store.saveRoom(room);
      send(response, 201, { target });
    } else if (request.method === 'POST' && action === 'sites') {
      const room = engine.room(gameId);
      const site = room.deploySite(await readJson(request));
      store.saveRoom(room);
      send(response, 201, { site });
    } else if (request.method === 'POST' && action === 'vehicle-reports') {
      const room = engine.room(gameId);
      const vehicle = room.reportVehicle(await readJson(request));
      store.saveRoom(room);
      send(response, 202, { vehicle });
    } else if (request.method === 'POST' && action === 'intercept') {
      const room = engine.room(gameId);
      const payload = await readJson(request);
      const result = room.commandIntercept(payload, {queueIfUntracked: payload.requireTracked !== true});
      if (result.accepted) store.saveRoom(room);
      send(response, result.accepted ? 202 : 409, result);
    } else {
      send(response, 405, { error: 'method-not-allowed' });
    }
  } catch (error) {
    send(response, 400, { error: error instanceof Error ? error.message : String(error) });
  }
});

server.listen(port, host, () => process.stdout.write(`atlantis simulation listening on http://${host}:${port}\n`));

function shutdown() {
  clearInterval(timer);
  clearInterval(checkpointTimer);
  store.saveEngine(engine);
  server.close(() => {
    store.close();
    process.exit(0);
  });
  setTimeout(() => process.exit(1), 3000).unref();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
