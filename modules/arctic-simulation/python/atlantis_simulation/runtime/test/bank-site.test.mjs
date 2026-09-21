import test from 'node:test';
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {SimulationRoom} from '../src/engine.mjs';
test('bank site preserves component UUIDs, avoids duplicate retries, restores ownership and exposes inventory',()=>{
 const room=new SimulationRoom('bank-test');
 const assets=Object.fromEntries(['radar','command','resupply','recovery','launcher-point-defense'].map(role=>[role,{id:randomUUID(),modelId:'defense-radar',ownerAccountId:randomUUID(),ownerUsername:'Owner'}]));
 const input={id:'banked',layerIds:['point-defense'],position:{x:1,y:2,z:3},bankAssets:assets,placementIntent:{key:'one'}};
 room.deploySite(input);room.deploySite(input);
 const snapshot=room.snapshot();
 const members=snapshot.vehicles.filter(v=>v.siteId==='banked');
 assert.equal(members.length,5);assert.deepEqual(new Set(members.map(v=>v.id)),new Set(Object.values(assets).map(a=>a.id)));
 assert.equal(snapshot.infrastructure.filter(v=>v.siteId==='banked').length,5);
 const restored=new SimulationRoom('bank-test',{},room.exportState());
 assert.deepEqual(restored.snapshot().vehicles,snapshot.vehicles);
 assert.deepEqual(restored.snapshot().sites.find(s=>s.id==='banked').bankAssets,assets);
 assert.throws(()=>room.deploySite({...input,placementIntent:{key:'changed'}}),/different terms/);
 assert.throws(()=>room.deploySite({...input,id:'duplicate-uuid-site'}),/Distinct bank UUIDs/);
});
