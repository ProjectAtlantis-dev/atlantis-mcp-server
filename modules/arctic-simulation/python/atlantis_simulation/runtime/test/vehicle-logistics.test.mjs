import assert from 'node:assert/strict';
import test from 'node:test';
import { VehicleLogisticsSystem } from '../src/vehicle-logistics.mjs';

test('a defense site creates individually tracked support vehicles', () => {
  const system = new VehicleLogisticsSystem();
  system.addSitePackage({
    id: 'site-1', readiness: 'deployed', position: { x: 1, y: 2, z: 3 },
    layers: [{ id: 'point-defense' }, { id: 'laser' }],
  });
  assert.equal(system.counts().total, 6);
  assert.equal(system.counts().deployed, 6);
  assert.equal(new Set(system.snapshot().map(vehicle => vehicle.id)).size, 6);
});

test('fuel reserve triggers maintenance and automated recovery', () => {
  const events = [];
  const system = new VehicleLogisticsSystem({ onEvent: (type, data) => events.push({ type, ...data }) });
  system.addSitePackage({ id: 'site-1', readiness: 'deployed', position: { x: 0, y: 0, z: 0 }, layers: [] });
  const radar = system.vehicles.get('site-1:radar');
  radar.fuel.liters = radar.fuel.reserveLiters;
  system.step(1);
  assert.equal(radar.status, 'maintenance');
  for (let index = 0; index < 120; index += 1) system.step(1);
  assert.equal(radar.status, 'deployed');
  assert.equal(radar.fuel.liters, radar.fuel.tankCapacityLiters);
  assert.ok(events.some(event => event.type === 'vehicle-maintenance-started'));
  assert.ok(events.some(event => event.type === 'vehicle-recovered'));
});

test('viewer position reports track one real vehicle and consume travel fuel', () => {
  const system = new VehicleLogisticsSystem();
  system.reportPosition({ id: 'player-amv', latitude: 64.18, longitude: -51.69, altitudeM: 20, headingDeg: 90 });
  const before = system.vehicles.get('player-amv').fuel.liters;
  const report = system.reportPosition({ id: 'player-amv', latitude: 64.181, longitude: -51.69, altitudeM: 21, headingDeg: 92 });
  assert.equal(system.counts().total, 1);
  assert.ok(report.fuel.liters < before);
  assert.equal(report.geodetic.latitude, 64.181);
});
