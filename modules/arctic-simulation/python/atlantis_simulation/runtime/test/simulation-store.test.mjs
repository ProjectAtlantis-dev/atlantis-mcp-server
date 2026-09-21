import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { SimulationEngine } from '../src/engine.mjs';
import { SimulationStore } from '../src/simulation-store.mjs';

function temporaryDatabase(t) {
  const directory = mkdtempSync(join(tmpdir(), 'atlantis-simulation-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  return join(directory, 'simulation.db');
}

test('a checkpoint restores authoritative room, vehicle, and event state', (t) => {
  const filename = temporaryDatabase(t);
  const firstStore = new SimulationStore(filename);
  const firstEngine = new SimulationEngine({ tickRateHz: 30 });
  const room = firstEngine.reset('campaign-1', { automaticDefense: false });
  room.spawnTarget({ id: 'shahed-persisted', start: { x: -1000, y: 0, z: 100 }, destination: { x: 0, y: 0, z: 0 }, speedMps: 50 });
  room.reportVehicle({ id: 'player-vehicle', latitude: 64.18, longitude: -51.69, altitudeM: 10 });
  room.step(0.5);
  firstStore.saveEngine(firstEngine);
  const expected = room.snapshot();
  firstStore.close();

  const secondStore = new SimulationStore(filename);
  const restoredEngine = new SimulationEngine({ tickRateHz: 30 });
  assert.equal(secondStore.restoreEngine(restoredEngine), 1);
  const restored = restoredEngine.room('campaign-1');
  assert.deepEqual(restored.snapshot(), expected);
  assert.ok(restored.eventsAfter(0).some(event => event.type === 'target-spawned'));
  assert.ok(restored.snapshot().vehicles.some(vehicle => vehicle.id === 'player-vehicle'));
  secondStore.close();
});

test('restored deterministic state continues with the same outcome', (t) => {
  const filename = temporaryDatabase(t);
  const store = new SimulationStore(filename);
  const originalEngine = new SimulationEngine({ tickRateHz: 30 });
  const original = originalEngine.reset('deterministic', {
    automaticDefense: false,
    sites: [{
      id: 'site', position: { x: 0, y: 0, z: 0 }, sensorRangeM: 5000,
      layers: [{ id: 'point', targetKinds: ['drone'], minRangeM: 0, maxRangeM: 5000, interceptorSpeedMps: 600, fuseRadiusM: 20, inventory: 2, effectiveness: 0.5 }],
    }],
  });
  original.spawnTarget({ id: 'target', start: { x: 1000, y: 0, z: 100 }, destination: { x: 0, y: 0, z: 0 }, speedMps: 20 });
  original.commandIntercept({ targetId: 'target' });
  original.step(0.25);
  store.saveRoom(original);

  const restoredEngine = new SimulationEngine({ tickRateHz: 30 });
  store.restoreEngine(restoredEngine);
  const restored = restoredEngine.room('deterministic');
  for (let index = 0; index < 20; index += 1) {
    original.step(0.1);
    restored.step(0.1);
  }
  assert.deepEqual(restored.snapshot(), original.snapshot());
  assert.deepEqual(restored.eventsAfter(0), original.eventsAfter(0));
  store.close();
});

test('resetting a game starts a new run without deleting the prior event ledger', (t) => {
  const store = new SimulationStore(temporaryDatabase(t));
  const engine = new SimulationEngine();
  const firstRun = engine.reset('scenario', { automaticDefense: false });
  firstRun.spawnTarget({ id: 'first-target' });
  store.saveRoom(firstRun);
  const firstRunId = firstRun.runId;
  const countStatement = store.database.prepare(
    'SELECT COUNT(*) AS count FROM simulation_event WHERE game_id = ? AND run_id = ?',
  );
  const firstRunEventCount = countStatement.get('scenario', firstRunId).count;

  const secondRun = engine.reset('scenario', { automaticDefense: false });
  store.saveRoom(secondRun);
  assert.notEqual(secondRun.runId, firstRunId);
  assert.equal(countStatement.get('scenario', firstRunId).count, firstRunEventCount);
  assert.ok(firstRunEventCount > 1);
  store.close();
});


test('terrain references persist once, refresh with a new patch, and restore exact verified samples',t=>{
 const store=new SimulationStore(temporaryDatabase(t)),engine=new SimulationEngine(),room=engine.room('terrain');
 const id='12345678-1234-4234-8234-123456789abc',ownerAccountId='22345678-1234-4234-8234-123456789abc';
 const surface={origin:{lat:64,lon:-51},minX:-10,minY:-10,stepM:2,rows:11,cols:11,heights:Array(121).fill(10),water:Array(121).fill(false)};
 room.groundControls.attach({id,ownerAccountId,definitionId:'patria-amv',terrainAssetId:'test',pose:{x:0,y:0,headingRad:0},surface});
 store.saveRoom(room);store.saveRoom(room);
 assert.equal(store.database.prepare('SELECT COUNT(*) AS n FROM simulation_terrain').get().n,1);
 const persisted=JSON.parse(store.database.prepare('SELECT state_json FROM simulation_room').get().state_json);
 assert.equal(persisted.storageVersion,2);assert.ok(persisted.controlledVehicles[0].surface.terrainRef);assert.equal(persisted.controlledVehicles[0].surface.heights,undefined);
 const vehicle=room.groundControls.get(id);vehicle.surface={...vehicle.surface,id:'new-patch',heights:Array(121).fill(10.5)};store.saveRoom(room);
 assert.equal(store.database.prepare('SELECT COUNT(*) AS n FROM simulation_terrain').get().n,2);
 const restored=new SimulationEngine();store.restoreEngine(restored);
 assert.deepEqual(restored.room('terrain').groundControls.get(id).surface,vehicle.surface);
 // Legacy embedded-terrain checkpoints remain readable.
 store.database.prepare('UPDATE simulation_room SET state_json=?').run(JSON.stringify(room.exportState()));
 const legacy=new SimulationEngine();store.restoreEngine(legacy);assert.deepEqual(legacy.room('terrain').groundControls.get(id).surface,vehicle.surface);
 store.saveRoom(room);store.database.prepare('DELETE FROM simulation_terrain').run();
 assert.throws(()=>store.restoreEngine(new SimulationEngine()),/grid is missing/);store.close();
});
