import test from 'node:test';
import assert from 'node:assert/strict';
import {facilityState,commandFacility,stepFacility} from '../src/facility-component.mjs';
import {InfrastructureState} from '../src/infrastructure-state.mjs';
test('facility access actions are versioned and interlocked until closure completes',()=>{
 const s=facilityState();commandFacility(s,{action:'facility_entry_outer',expectedRevision:0});
 assert.throws(()=>commandFacility(s,{action:'facility_freight_open',expectedRevision:1}),/Close/);
 stepFacility(s,1);assert.equal(s.entry.outer,.5);
 commandFacility(s,{action:'facility_entry_close',expectedRevision:1});
 assert.throws(()=>commandFacility(s,{action:'facility_freight_open',expectedRevision:2}),/Close/);
 stepFacility(s,1);commandFacility(s,{action:'facility_freight_open',expectedRevision:2});stepFacility(s,1);assert.equal(s.freight,.5);
 assert.throws(()=>commandFacility(s,{action:'facility_entry_inner',expectedRevision:3}),/Close/);
 assert.throws(()=>commandFacility(s,{action:'facility_freight_close',expectedRevision:2}),/Stale/);
});
test('saved facility state resumes without snapping; placement cannot inject it',()=>{
 const state=new InfrastructureState(),entity=state.place({modelId:'future-city-water-plant',position:{x:0,y:0,z:0},componentState:{freight:1}});
 assert.equal(entity.componentState.freight,0);
 state.command({id:entity.id,action:'facility_freight_open',expectedRevision:0});state.step(.5);
 const restored=new InfrastructureState(JSON.parse(JSON.stringify(state.snapshot())));assert.deepEqual(restored.snapshot(),state.snapshot());
 restored.step(1.5);assert.equal(restored.snapshot()[0].componentState.freight,1);
 const corrupt=state.snapshot();corrupt[0].componentState.entry.outer=1;assert.throws(()=>new InfrastructureState(corrupt),/Conflicting/);
});
