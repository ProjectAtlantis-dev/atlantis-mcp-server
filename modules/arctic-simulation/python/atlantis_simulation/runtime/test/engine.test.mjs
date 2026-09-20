import assert from 'node:assert/strict';
import test from 'node:test';
import { SimulationRoom } from '../src/engine.mjs';

test('fixed steps move a Shahed on authoritative state', () => {
  const room = new SimulationRoom('proof', { automaticDefense: false, sites: [{ id: 'site', position: { x: 0, y: 0, z: 0 }, sensorRangeM: 10000, layers: [] }] });
  const target = room.spawnTarget({ id: 'shahed-1', start: { x: -1000, y: 0, z: 100 }, destination: { x: 0, y: 0, z: 0 }, speedMps: 100 });
  room.step(0.5);
  const moved = room.snapshot().targets[0];
  assert.equal(room.tick, 1);
  assert.ok(moved.position.x > target.position.x);
  assert.equal(moved.status, 'active');
});

test('strategic intercept selects a ready layer and resolves underneath it', () => {
  const room = new SimulationRoom('proof', {
    automaticDefense: false,
    sites: [{ id: 'site', position: { x: 0, y: 0, z: 0 }, sensorRangeM: 10000, layers: [{ id: 'point-defense', targetKinds: ['drone'], minRangeM: 0, maxRangeM: 10000, interceptorSpeedMps: 1000, fuseRadiusM: 10, reloadSeconds: 1, inventory: 2, effectiveness: 1 }] }],
  });
  room.spawnTarget({ id: 'shahed-1', start: { x: 500, y: 0, z: 50 }, destination: { x: 0, y: 0, z: 0 }, speedMps: 20 });
  const command = room.commandIntercept({ targetId: 'shahed-1' });
  assert.equal(command.accepted, true);
  for (let index = 0; index < 30 && room.statistics.intercepted === 0; index += 1) room.step(0.1);
  const snapshot = room.snapshot();
  assert.equal(snapshot.targets[0].status, 'intercepted');
  assert.equal(snapshot.statistics.intercepted, 1);
  assert.equal(snapshot.counts.targets.intercepted, 1);
  assert.ok(room.eventsAfter(0).some(event => event.type === 'target-intercepted'));
});

test('automatic defense engages tracked targets without viewer ticks', () => {
  const room = new SimulationRoom('auto', {
    automaticDefense: true,
    sites: [{ id: 'site', position: { x: 0, y: 0, z: 0 }, sensorRangeM: 5000, layers: [{ id: 'layer', targetKinds: ['drone'], minRangeM: 0, maxRangeM: 5000, interceptorSpeedMps: 500, fuseRadiusM: 20, inventory: 1, effectiveness: 1 }] }],
  });
  room.spawnTarget({ start: { x: 1000, y: 0, z: 100 }, destination: { x: 0, y: 0, z: 0 }, speedMps: 40 });
  for (let index = 0; index < 12; index += 1) room.step(1 / 30);
  assert.equal(room.statistics.launched, 1);
  assert.equal(room.snapshot().counts.vehicles.engaged, 1);
});

test('deployed layers remain offline while construction is incomplete', () => {
  const room = new SimulationRoom('build', { automaticDefense: false });
  const site = room.deploySite({ id: 'forward-site', position: { x: -500, y: 0, z: 0 }, layerIds: ['point-defense'], buildSeconds: 2 });
  assert.equal(site.readiness, 'constructing');
  room.step(1);
  assert.equal(room.snapshot().sites.find(item => item.id === 'forward-site').readiness, 'constructing');
  room.step(1);
  const snapshot = room.snapshot();
  assert.equal(snapshot.sites.find(item => item.id === 'forward-site').readiness, 'deployed');
  assert.equal(snapshot.counts.defense.constructingSites, 0);
  assert.ok(room.eventsAfter(0).some(event => event.type === 'defense-site-deployed'));
});

test('ballistic targets follow a server-owned arc', () => {
  const room = new SimulationRoom('ballistic', { automaticDefense: false });
  room.spawnTarget({ id: 'missile-1', kind: 'ballistic', start: { x: -10000, y: 0, z: 100 }, destination: { x: 0, y: 0, z: 0 }, durationSeconds: 20, apexM: 5000 });
  for (let index = 0; index < 10; index += 1) room.step(1);
  const target = room.snapshot().targets[0];
  assert.equal(target.trajectory, 'ballistic');
  assert.ok(target.position.z > 5000);
  assert.ok(target.position.x > -10000 && target.position.x < 0);
});

test('directed energy requires continuous server-side dwell', () => {
  const room = new SimulationRoom('laser', { automaticDefense: false });
  room.spawnTarget({ id: 'drone-1', start: { x: 1500, y: -450, z: 200 }, destination: { x: 0, y: 0, z: 0 }, speedMps: 10 });
  for (let index = 0; index < 11; index += 1) room.step(0.05);
  const command = room.commandIntercept({ targetId: 'drone-1', layerId: 'directed-energy' });
  assert.equal(command.accepted, true);
  room.step(1);
  assert.equal(room.snapshot().targets[0].status, 'active');
  room.step(0.5);
  assert.equal(room.snapshot().targets[0].status, 'intercepted');
});
