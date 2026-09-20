import test from 'node:test';
import assert from 'node:assert/strict';
import {InfrastructureState} from '../src/infrastructure-state.mjs';
import {SimulationRoom} from '../src/engine.mjs';
test('placement validates IDs and coordinates, updates atomically, and returns detached snapshots',()=>{
 const s=new InfrastructureState();const a=s.place({id:'tower',modelId:'network-radome-tower',position:{x:1,y:2,z:3}});
 a.position.x=999;assert.equal(s.snapshot()[0].position.x,1);
 assert.throws(()=>s.place({id:'tower',modelId:a.modelId,position:a.position}),/duplicate/);
 assert.throws(()=>s.move({id:'tower',position:{x:NaN,y:2,z:3}}),/finite/);
 assert.equal(s.snapshot()[0].position.x,1);
 assert.equal(s.move({id:'tower',position:{x:4,y:5,z:6},headingDeg:370}).headingDeg,10);
 assert.throws(()=>s.place({modelId:'not-a-model',position:{x:0,y:0,z:0}}),/unknown/);
 s.remove({id:'tower'});assert.deepEqual(s.snapshot(),[]);
});
test('infrastructure placements survive the existing room checkpoint contract',()=>{
 const room=new SimulationRoom('infrastructure-test');room.infrastructure.place({id:'radome',modelId:'network-radome',position:{x:20,y:30,z:4},headingDeg:90});
 const restored=new SimulationRoom(room.id,{},JSON.parse(JSON.stringify(room.exportState())));
 assert.deepEqual(restored.snapshot().infrastructure,room.snapshot().infrastructure);
 const legacy=room.exportState();delete legacy.infrastructure;assert.deepEqual(new SimulationRoom(room.id,{},legacy).snapshot().infrastructure,[]);
});
test('bound support visuals follow authoritative position and logistics, not placement edits',()=>{
 const room=new SimulationRoom('bound-infrastructure'),vehicle=[...room.vehicleSystem.vehicles.values()].find(v=>v.role==='logistics');
 room.infrastructure.place({id:'cargo',modelId:'support-resupply',sourceVehicleId:vehicle.id,position:{x:99,y:99,z:99}});
 vehicle.position={x:1,y:2,z:3};vehicle.status='maintenance';vehicle.fuel.liters=10;
 const entity=room.snapshot().infrastructure[0];assert.deepEqual(entity.position,vehicle.position);assert.equal(entity.operationalState.status,'maintenance');
 assert.equal(entity.operationalState.fuel.liters,10);assert.throws(()=>room.infrastructure.move({id:'cargo',position:{x:0,y:0,z:0}}),/follows/);
 entity.operationalState.fuel.liters=999;assert.equal(vehicle.fuel.liters,10);
});
